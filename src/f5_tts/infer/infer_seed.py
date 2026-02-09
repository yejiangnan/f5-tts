import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from multiprocessing import Process, cpu_count, get_context
from pathlib import Path
from tqdm import tqdm

import numpy as np
import soundfile as sf
import tomli
import torch
import torchaudio
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

from f5_tts.model.utils import seed_everything
from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    speed,
    sway_sampling_coef,
    target_rms,
)


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--no_legacy_text",
    action="store_false",
    help="Not to use lossy ASCII transliterations of unicode text in saved file names.",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--gen_duration",
    type=float,
    help="Fix the generated audio duration (excluding ref audio) in seconds. If set, will override fix_duration.",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on",
)
parser.add_argument(
    "--base_output_dir",
    type=str,
    help="The path to base output directory",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=None,
    help="Number of worker processes for parallel generation. Default: number of GPUs if available, else number of CPU cores.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible generation. If not specified, uses random seed for each generation.",
)

args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parameters

model = args.model or config.get("model", "F5TTS_v1_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
ref_text = (
    args.ref_text
    if args.ref_text is not None
    else config.get("ref_text", "Some call me nature, others call me mother nature.")
)
gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
use_legacy_text = args.no_legacy_text or config.get("no_legacy_text", False)  # no_legacy_text is a store_false arg
if save_chunk and use_legacy_text:
    print(
        "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
    )

remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
gen_duration = args.gen_duration if args.gen_duration is not None else config.get("gen_duration", None)
seed = args.seed if args.seed is not None else config.get("seed", None)
device = args.device or config.get("device", device)


# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))


# ignore gen_text if gen_file provided

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()


# output path

wave_path = Path(output_dir) / output_file
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# Model configuration (needed for worker processes)
model_cfg_path = args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
model_cfg = OmegaConf.load(model_cfg_path)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch

repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

if model != "F5TTS_Base":
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

# override for previous models
if model == "F5TTS_Base":
    if vocoder_name == "vocos":
        ckpt_step = 1200000
    elif vocoder_name == "bigvgan":
        model = "F5TTS_Base_bigvgan"
        ckpt_type = "pt"
elif model == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

if not ckpt_file:
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))

# Vocoder local path
if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
else:
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"


# inference process


def worker_process(worker_id, tasks_chunk, config):
    """工作进程：加载模型后循环处理分配给自己的任务块"""
    # 获取seed配置
    seed = config.get("seed")
    if seed is not None:
        print(f"[进程 {worker_id}] 将使用固定随机种子: {seed}")
    
    # 确定使用的 GPU
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        gpu_id = worker_id % num_gpus
        process_device = f"cuda:{gpu_id}"
        device_str = f"GPU {gpu_id}"
    else:
        gpu_id = None
        process_device = device
        device_str = "CPU"
    
    print(f"[进程 {worker_id} {device_str}] 开始加载模型...")
    
    # 加载 vocoder
    vocoder_name = config["vocoder_name"]
    load_vocoder_from_local = config["load_vocoder_from_local"]
    vocoder_local_path = config["vocoder_local_path"]
    
    vocoder = load_vocoder(
        vocoder_name=vocoder_name,
        is_local=load_vocoder_from_local,
        local_path=vocoder_local_path,
        device=process_device,
    )
    
    # 加载 TTS 模型
    model = config["model"]
    ckpt_file = config["ckpt_file"]
    vocab_file = config["vocab_file"]
    model_cfg_path = config["model_cfg_path"]
    
    # 重新加载模型配置（因为无法序列化传递）
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    
    if model_cls.__name__ == "DiT":
        if "emphasis_enhanced" not in model_arc:
            model_arc["emphasis_enhanced"] = "transfomer"
        elif model_arc.get("emphasis_enhanced") != "transfomer":
            model_arc["emphasis_enhanced"] = "transfomer"
    
    ema_model = load_model(
        model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=process_device
    )
    
    print(f"[进程 {worker_id} {device_str}] 模型加载完成，开始处理 {len(tasks_chunk)} 个任务")
    
    # 循环处理分配给自己的任务
    output_dir = config["output_dir"]
    gen_duration = config["gen_duration"]
    fix_duration = config["fix_duration"]
    
    success_count = 0
    fail_count = 0
    
    # 创建该进程的进度条
    pbar = tqdm(total=len(tasks_chunk), desc=f"进程 {worker_id} ({device_str})", position=worker_id, leave=True)
    
    for utt, ref_text, ref_audio, gt_text in tasks_chunk:
        try:
            # 如果指定了seed，为每个任务设置相同的seed（确保可复现）
            # 如果没有指定seed，使用随机seed
            task_seed = seed
            if task_seed is None:
                import random
                task_seed = random.randint(0, 2**31 - 1)
            seed_everything(task_seed)
            
            # If gen_duration is specified, calculate fix_duration from reference audio duration + gen_duration
            current_fix_duration = fix_duration
            if gen_duration is not None:
                # Load reference audio to get its duration
                ref_audio_data, ref_sr = torchaudio.load(ref_audio)
                ref_audio_duration_sec = ref_audio_data.shape[-1] / ref_sr
                current_fix_duration = ref_audio_duration_sec + gen_duration
            
            audio_segment, final_sample_rate, spectrogram = infer_process(
                ref_audio,
                ref_text,
                gt_text,
                ema_model,
                vocoder,
                mel_spec_type=vocoder_name,
                target_rms=config["target_rms"],
                cross_fade_duration=config["cross_fade_duration"],
                nfe_step=config["nfe_step"],
                cfg_strength=config["cfg_strength"],
                sway_sampling_coef=config["sway_sampling_coef"],
                speed=config["speed"],
                fix_duration=current_fix_duration,
                device=process_device,
            )
            
            wave_path = os.path.join(output_dir, f"{utt}.wav")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            sf.write(wave_path, audio_segment, final_sample_rate)
            
            # Remove silence if needed
            if config["remove_silence"]:
                remove_silence_for_generated_wav(wave_path)
            
            success_count += 1
            pbar.update(1)
            pbar.set_postfix({"成功": success_count, "失败": fail_count})
        except Exception as e:
            fail_count += 1
            print(f"\n[进程 {worker_id}] 处理 {utt} 时出错: {e}")
            pbar.update(1)
            pbar.set_postfix({"成功": success_count, "失败": fail_count})
    
    pbar.close()
    print(f"[进程 {worker_id} {device_str}] 完成！成功: {success_count}, 失败: {fail_count}")


def main():
    input_dir = "/data/seed-tts-eval/seedtts_testset/zh/meta.lst"
    output_dir = args.base_output_dir
    if not output_dir:
        output_dir = "/data/seed-tts-eval/outputs/four_sec"
    
    with open(input_dir, "r") as f:
        gen_texts = f.readlines()

    data = []
    for gen_text in gen_texts:
        utt, prompt_text, prompt_wav, gt_text = gen_text.strip().split("|")
        prompt_wav = f"/data/seed-tts-eval/seedtts_testset/zh/{prompt_wav}"
        data.append((utt, prompt_text, prompt_wav, gt_text))
    
    # 确定使用的进程数
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.num_workers is not None:
        num_workers = args.num_workers
    elif num_gpus > 0:
        num_workers = num_gpus
    else:
        num_workers = min(cpu_count(), len(data))
    
    print(f"使用 {num_workers} 个进程进行并行生成（{'GPU' if num_gpus > 0 else 'CPU'}模式）")
    print(f"总任务数: {len(data)}")
    
    # 检查已存在的文件（断点恢复）
    if os.path.exists(output_dir):
        existing_files = set()
        for wav_file in Path(output_dir).glob("*.wav"):
            existing_files.add(wav_file.stem)
        
        if existing_files:
            print(f"发现 {len(existing_files)} 个已存在的文件，将跳过这些任务（断点恢复）")
            original_count = len(data)
            data = [(utt, ref_text, ref_audio, gt_text) for utt, ref_text, ref_audio, gt_text in data if utt not in existing_files]
            skipped_count = original_count - len(data)
            print(f"跳过 {skipped_count} 个已完成的任务，剩余 {len(data)} 个任务需要处理")
    
    if len(data) == 0:
        print("所有任务已完成，无需处理！")
        return
    
    # 准备共享配置
    shared_config = {
        "model": model,
        "ckpt_file": ckpt_file,
        "vocab_file": vocab_file,
        "vocoder_name": vocoder_name,
        "load_vocoder_from_local": load_vocoder_from_local,
        "vocoder_local_path": vocoder_local_path,
        "output_dir": output_dir,
        "gen_duration": gen_duration,
        "fix_duration": fix_duration,
        "target_rms": target_rms,
        "cross_fade_duration": cross_fade_duration,
        "nfe_step": nfe_step,
        "cfg_strength": cfg_strength,
        "sway_sampling_coef": sway_sampling_coef,
        "speed": speed,
        "remove_silence": remove_silence,
        "model_cfg_path": model_cfg_path,
        "seed": seed,
    }
    
    # 将任务列表切成 num_workers 份
    chunk_size = len(data) // num_workers
    task_chunks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        if i == num_workers - 1:  # 最后一个进程处理剩余的所有任务
            end_idx = len(data)
        else:
            end_idx = (i + 1) * chunk_size
        task_chunks.append(data[start_idx:end_idx])
    
    print(f"任务分配：")
    for i, chunk in enumerate(task_chunks):
        print(f"  进程 {i}: {len(chunk)} 个任务")
    
    # 使用多进程处理。CUDA 不能在 fork 的子进程中重初始化，必须用 spawn 启动子进程
    ctx = get_context("spawn")
    processes = []
    for worker_id in range(num_workers):
        p = ctx.Process(target=worker_process, args=(worker_id, task_chunks[worker_id], shared_config))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 统计实际生成的文件数
    if os.path.exists(output_dir):
        actual_wav_count = len(list(Path(output_dir).glob("*.wav")))
        print(f"\n实际生成的文件数: {actual_wav_count}")
    
    print(f"\n所有进程完成！预期生成 {len(data)} 个音频文件。")



if __name__ == "__main__":
    main()
