import argparse
import os
import re
import shutil
import json
from subprocess import run
import torchaudio
import soundfile as sf
from tqdm import tqdm
from pprint import pprint
import multiprocessing




def mfa(wavs, annotation, num_jobs=None, batch_size=10000):
    """
    分批运行 MFA 对齐命令
    
    Args:
        wavs: 音频文件目录
        annotation: 输出标注目录
        num_jobs: 并行进程数，如果为 None 则根据 CPU 核心数和文件数量自动调整
        batch_size: 每批处理的文件数量，默认 10000
    """
    from pathlib import Path
    import tempfile
    import shutil
    
    wav_path = Path(wavs)
    annotation_path = Path(annotation)
    annotation_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 wav 文件
    all_wav_files = sorted(list(wav_path.glob("*.wav")))
    total_files = len(all_wav_files)
    
    if total_files == 0:
        print(f"[Warning] 在 {wavs} 中没有找到 wav 文件")
        return
    
    print(f"[Info] 总共找到 {total_files} 个文件，将分批处理，每批 {batch_size} 个文件")
    
    # 计算批次数
    num_batches = (total_files + batch_size - 1) // batch_size
    
    if num_jobs is None:
        cpu_count = multiprocessing.cpu_count()
        # 对于每批 1 万文件，使用合理的进程数
        # num_jobs = min(cpu_count, max(8, min(32, batch_size // 500)))
        num_jobs = cpu_count
        print(f"[Info] 自动设置每批进程数为 {num_jobs}")
    
    # 检查已完成的批次（断点续传功能）
    completed_batches = set()
    if annotation_path.exists():
        # 统计已存在的 TextGrid 文件，判断哪些批次已完成
        existing_textgrids = set(f.stem for f in annotation_path.glob("*.TextGrid"))
        print(f"[Info] 检测到输出目录中已有 {len(existing_textgrids)} 个对齐文件")
        
        # 检查每个批次是否已完成（通过检查该批次的所有文件是否都存在）
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            batch_files = all_wav_files[start_idx:end_idx]
            
            # 检查该批次的所有文件是否都已对齐
            batch_complete = True
            for wav_file in batch_files:
                if wav_file.stem not in existing_textgrids:
                    batch_complete = False
                    break
            
            if batch_complete:
                completed_batches.add(batch_idx)
                print(f"[Info] 批次 {batch_idx + 1}/{num_batches} 已完成，将跳过")
    
    if completed_batches:
        print(f"[Info] 共 {len(completed_batches)} 个批次已完成，将从批次 {len(completed_batches) + 1} 开始处理")
    
    # 记录成功和失败的批次
    successful_batches = []
    failed_batches = []
    
    # 分批处理
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_files = all_wav_files[start_idx:end_idx]
        batch_num = batch_idx + 1
        
        # 跳过已完成的批次
        if batch_idx in completed_batches:
            successful_batches.append(batch_idx)
            continue
        
        print(f"\n[Info] 处理第 {batch_num}/{num_batches} 批（文件 {start_idx+1}-{end_idx}，共 {len(batch_files)} 个文件）")
        
        # 创建临时目录存放当前批次的文件
        with tempfile.TemporaryDirectory(prefix=f"mfa_batch_{batch_idx}_") as temp_wav_dir:
            temp_wav_path = Path(temp_wav_dir)
            
            # 复制当前批次的 wav 文件和对应的 .lab 文件到临时目录
            print(f"[Info] 复制文件到临时目录...")
            for wav_file in tqdm(batch_files, desc=f"复制批次 {batch_num}"):
                # 复制 wav 文件
                shutil.copy2(wav_file, temp_wav_path / wav_file.name)
                # 复制对应的 .lab 文件（如果存在）
                lab_file = wav_file.with_suffix('.lab')
                if lab_file.exists():
                    shutil.copy2(lab_file, temp_wav_path / lab_file.name)
                else:
                    print(f"[Warning] 未找到对应的 .lab 文件: {lab_file}")
            
            # 创建临时输出目录
            temp_annotation_dir = tempfile.mkdtemp(prefix=f"mfa_annotation_{batch_idx}_")
            temp_annotation_path = Path(temp_annotation_dir)
            
            try:
                # 运行 MFA 对齐
                # 注意：MFA 会在 /home/i-yejiangnan/Documents/MFA/ 目录下创建临时文件
                # 如果磁盘空间不足，需要清理该目录下的旧文件
                cmd = f"mfa align {temp_wav_path} mandarin_china_mfa mandarin_mfa {temp_annotation_path} --clean -j {num_jobs} --single_speaker"
                print(f"[Info] 运行 MFA 对齐（批次 {batch_num}/{num_batches}）...")
                run(cmd, shell=True, check=True)
                
                # 将结果复制到最终输出目录
                print(f"[Info] 复制对齐结果到输出目录...")
                copied_count = 0
                for textgrid_file in temp_annotation_path.glob("*.TextGrid"):
                    shutil.copy2(textgrid_file, annotation_path / textgrid_file.name)
                    copied_count += 1
                
                if copied_count == 0:
                    print(f"[Warning] 批次 {batch_num}/{num_batches} 没有生成任何 TextGrid 文件")
                    failed_batches.append(batch_idx)
                else:
                    print(f"[Info] 批次 {batch_num}/{num_batches} 完成，共复制 {copied_count} 个文件")
                    successful_batches.append(batch_idx)
                
            except Exception as e:
                print(f"[Error] 批次 {batch_num}/{num_batches} 处理失败: {e}")
                print(f"[Error] 错误详情：{type(e).__name__}: {str(e)}")
                failed_batches.append(batch_idx)
                # 不抛出异常，继续处理后续批次
                continue
            finally:
                # 清理临时标注目录
                if temp_annotation_path.exists():
                    shutil.rmtree(temp_annotation_path, ignore_errors=True)
    
    # 输出处理结果统计
    print(f"\n{'='*60}")
    print(f"[Info] 处理完成统计：")
    print(f"  总批次数: {num_batches}")
    print(f"  成功批次: {len(successful_batches)}")
    print(f"  失败批次: {len(failed_batches)}")
    
    if successful_batches:
        print(f"  成功的批次编号: {sorted([b+1 for b in successful_batches])}")
    
    if failed_batches:
        print(f"  失败的批次编号: {sorted([b+1 for b in failed_batches])}")
        print(f"\n[Warning] 有 {len(failed_batches)} 个批次处理失败，可以重新运行脚本继续处理失败的批次")
    
    # 统计最终输出的文件数
    final_textgrid_count = len(list(annotation_path.glob("*.TextGrid"))) if annotation_path.exists() else 0
    print(f"\n[Info] 最终输出目录中共有 {final_textgrid_count} 个对齐文件（目标: {total_files} 个）")
    print(f"{'='*60}")
    
    # 如果有失败的批次，返回非零退出码（但不抛出异常，让用户可以重新运行）
    if failed_batches:
        print(f"\n[Info] 可以重新运行脚本继续处理失败的批次")
        return False
    else:
        print(f"\n[Info] 所有批次处理完成！共处理 {total_files} 个文件，输出到 {annotation_path}")
        return True

def get_prominence(wavs, annotation, json_output, wav_out_dir, num_workers=None):
    """
    运行 prominence 计算
    
    Args:
        wavs: 音频文件目录
        annotation: 标注文件目录
        json_output: 输出 JSON 文件路径
        wav_out_dir: 输出 WAV 文件目录
        num_workers: 并行进程数，如果为 None 则使用所有可用 CPU 核心
    """
    import time
    from pathlib import Path
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # 统计文件数量
    wav_path = Path(wavs)
    annotation_path = Path(annotation)
    wav_count = len(list(wav_path.glob("*.wav"))) if wav_path.exists() else 0
    textgrid_count = len(list(annotation_path.glob("*.TextGrid"))) if annotation_path.exists() else 0
    
    print(f"[Info] 准备计算 prominence:")
    print(f"  音频文件数: {wav_count}")
    print(f"  标注文件数: {textgrid_count}")
    print(f"  使用进程数: {num_workers} (CPU 核心数: {multiprocessing.cpu_count()})")
    
    start_time = time.time()
    
    # 直接调用 Python 脚本，传递 num_workers 参数
    cmd = [
        "python", 
        "wavelet_prosody_toolkit/compute_prominence.py",
        "--wav_dir", str(wavs),
        "--annotation_dir", str(annotation),
        "--json_output", str(json_output),
        "--wav_out_dir", str(wav_out_dir),
        "--num_workers", str(num_workers) 
    ]
    print(f"[Info] 开始 prominence 计算...")
    run(cmd)
    
    elapsed_time = time.time() - start_time
    if wav_count > 0:
        avg_time_per_file = elapsed_time / wav_count
        print(f"\n[Info] Prominence 计算完成！")
        print(f"  总耗时: {elapsed_time/60:.1f} 分钟 ({elapsed_time:.1f} 秒)")
        print(f"  平均每个文件: {avg_time_per_file:.2f} 秒")
        print(f"  处理速度: {wav_count/elapsed_time:.1f} 文件/秒")

def pick_prominence(proc_dir, json_output, raw_text):
    metadata = {}
    with open(raw_text, "r") as f:
        for idx, line in enumerate(f):
            # 使用正则表达式提取 <strong></strong> 之间的词
            strong_matches = re.findall(r'<strong>(.*?)</strong>', line)
            if not strong_matches:
                # 跳过没有 <strong> 标签的行，避免匹配错误
                print(f"Warning: 行 {idx} 没有 <strong> 标签，已跳过: {line[:50]}...")
                continue
            strong_words = strong_matches[0]
            uttid = f"{idx:05d}"
            metadata[uttid] = {"emphasis": strong_words, "info": [], "text": line.strip()}
    
    with open(json_output, "r") as f:
        data = json.load(f)

    for key, value in data.items():
        uttid = key.split("_")[0]
        # 跳过 metadata 中不存在的 uttid（对应没有 strong 标签的文本行）
        if uttid not in metadata:
            print(f"Warning: 音频文件 {key} 对应的文本行 {uttid} 没有 <strong> 标签，已跳过")
            continue
        emphasis_word = metadata[uttid]['emphasis']
        words = value['words']
        # 对于每个样本，找到所有匹配的词中 prominence 值最高的那个
        max_prominence = None
        for word in words:
            w, v = word.split(":")
            v = float(v)
            if w in emphasis_word or emphasis_word in w:
                if max_prominence is None or v > max_prominence:
                    max_prominence = v
        # 只有当找到匹配的词时，才添加一次（每个样本只添加一次）
        if max_prominence is not None:
            metadata[uttid]["info"].append((key, max_prominence))
    
    # 在所有数据收集完成后，对每个 uttid 的 info 进行排序（只排序一次）
    for uttid in metadata.keys():
        metadata[uttid]["info"].sort(key=lambda x: x[1])
    
    dpo_metadata = []
    
    for key, value in tqdm(metadata.items(), total=len(metadata), desc="Processing metadata"):
        if len(value["info"]) == 0:
            continue
        best_key = value["info"][-1][0]
        best_wav_path = f"{proc_dir}/wavs/{best_key}.wav"
        best_dest_path = f"{proc_dir}/emphasis_wavs/{best_key}.wav"
        shutil.copy(best_wav_path, best_dest_path)
        worst_key = value["info"][0][0]
        worst_wav_path = f"{proc_dir}/wavs/{worst_key}.wav"
        worst_dest_path = f"{proc_dir}/non_emphasis_wavs/{worst_key}.wav"
        shutil.copy(worst_wav_path, worst_dest_path)

        # 计算两个wav文件的时长（秒，保留一位小数）
        try:
            # 使用 soundfile.info() 获取音频信息，不需要加载整个音频文件
            best_info = sf.info(best_wav_path)
            best_duration_sec = round(best_info.frames / best_info.samplerate, 1)
            
            worst_info = sf.info(worst_wav_path)
            worst_duration_sec = round(worst_info.frames / worst_info.samplerate, 1)
            
            dpo_metadata.append({
                "best_wav_path": best_wav_path,
                "worst_wav_path": worst_wav_path,
                "best_duration_sec": best_duration_sec,
                "worst_duration_sec": worst_duration_sec,
                "text": value["text"]
            })
        except Exception as e:
            print(f"  Warning: Failed to calculate audio durations: {e}")

    with open(f"/data/F5-TTS/{proc_dir}/dpo_metadata.json", "w") as f:
        json.dump(dpo_metadata, f, ensure_ascii=False, indent=4)

    with open(f"/data/F5-TTS/{proc_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    proc_dir = "stepf06_data/batch_1"
    raw_text = "/data/F5-TTS/text_file/emphasis_text_format_batch6_f06.txt"
    # proc_dir = "stepf15_data/batch_1"
    # raw_text = "/data/F5-TTS/text_file/emphasis_text_format_batch7_f15.txt"
    os.makedirs(f"/data/F5-TTS/{proc_dir}/non_emphasis_wavs", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wavs", 
        default=f"/data/F5-TTS/{proc_dir}/wavs",
        type=str, 
        required=False
    )
    parser.add_argument(
        "--annotation", 
        default=f"/data/F5-TTS/{proc_dir}/aligned",
        type=str, 
        required=False
    )
    parser.add_argument(
        "--json_output", 
        default=f"/data/F5-TTS/{proc_dir}/prominence.json",
        type=str, 
        required=False
    )
    parser.add_argument(
        "--wav_out_dir", 
        default=f"/data/F5-TTS/{proc_dir}/emphasis_wavs",
        type=str, 
        required=False
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=None,
        help="MFA 对齐使用的并行进程数（默认：使用所有 CPU 核心）"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Prominence 计算使用的并行进程数（默认：使用所有 CPU 核心）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="MFA 对齐每批处理的文件数量（默认：10000）"
    )

    args = parser.parse_args()
    # mfa(args.wavs, args.annotation, num_jobs=args.num_jobs, batch_size=args.batch_size)
    # get_prominence(args.wavs, args.annotation, args.json_output, args.wav_out_dir, num_workers=args.num_workers)
    pick_prominence(proc_dir, args.json_output, raw_text)

    
