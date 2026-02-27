from __future__ import annotations

import copy
import gc
import json
import math
import os
import re
import string
import sys
import shutil
from pathlib import Path
from subprocess import run, DEVNULL, PIPE, CalledProcessError
from pprint import pprint

# Set NCCL timeout before importing torch
if "NCCL_TIMEOUT" not in os.environ:
    os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes (in seconds)
if "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

import torch
import torchaudio
import wandb
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset 
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.grpo_dataset import grpo_collate_fn, DistributedKRepeatSampler
from f5_tts.model.utils import default, exists
from f5_tts.model.trainer import Trainer
from collections import defaultdict
import numpy as np
import hashlib
import random



class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: CFM,
        # 采样相关参数
        num_samples_per_prompt: int = 6,  # k值，每个prompt采样k个样本
        sample_batch_size: int = 64,  # 采样时的batch size
        # num_inference_steps: int = 40, 
        timestep_fraction: float = 1.0,  # 训练时使用的timestep比例 (0-1)
        noise_level: float = 0.1,  # 噪声级别，0.0 表示确定性采样
        same_latent: bool = False,  # 是否使用相同的初始latent
        # 训练相关参数
        num_inner_epochs: int = 1,  # 内层epoch数
        beta: float = 0.1,  # KL 散度权重，建议 0.01~0.1；过大(如10)会导致 KL 快速趋近 0、策略坍缩到 ref、生成退化/噪音
        adv_clip_max: float = 5.0,  # advantage裁剪最大值
        clip_range: float = 1e-4,  # PPO clip 范围，建议 0.1~0.2；过小(如1e-4)会导致 clipfrac≈1、policy_loss 无效
        # 统计跟踪相关参数
        per_prompt_stat_tracking: bool = True,  # 是否使用per-prompt统计跟踪
        global_std: bool = True,  # 是否使用全局标准差（用于统计跟踪）
        # 奖励函数相关参数
        reward_fn: dict = None,  # 奖励函数配置
        **kwargs,  # 其他参数传给父类Trainer
    ):
        super().__init__(model, **kwargs)
        print(f"GRPO gradient_accumulation_steps: {self.accelerator.gradient_accumulation_steps}")
        
        # 保存GRPO特有参数
        self.num_samples_per_prompt = num_samples_per_prompt
        self.sample_batch_size = sample_batch_size
        # self.num_inference_steps = num_inference_steps
        self.timestep_fraction = timestep_fraction
        self.noise_level = noise_level
        self.same_latent = same_latent
        
        self.num_inner_epochs = num_inner_epochs
        self.beta = beta
        self.adv_clip_max = adv_clip_max
        self.clip_range = clip_range
        
        self.per_prompt_stat_tracking = per_prompt_stat_tracking
        self.global_std = global_std
        
        self.reward_fn = reward_fn

        
        # 计算实际训练的timestep数量
        # self.num_train_timesteps = int(self.num_inference_steps * self.timestep_fraction)
        
        # 如果每个prompt只采样1个样本，则禁用per-prompt统计跟踪
        if self.num_samples_per_prompt == 1:
            self.per_prompt_stat_tracking = False

        import copy
        self.ref_model = copy.deepcopy(self.accelerator.unwrap_model(self.model))
        self.ref_model.to(self.accelerator.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # GRPO 训练时关闭 dropout，使 policy 与 ref 前向一致，初始 KL≈0
        for m in self.accelerator.unwrap_model(self.model).modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0

        # 初始化统计跟踪器
        if self.per_prompt_stat_tracking:
            try:
                from f5_tts.model.stat_tracking import PerPromptStatTracker  # type: ignore
                self.stat_tracker = PerPromptStatTracker(global_std=self.global_std)
            except ImportError:
                # 如果stat_tracking模块不存在，使用None
                self.stat_tracker = None
                self.per_prompt_stat_tracking = False
        else:
            self.stat_tracker = None
        
        # 初始化vocoder（如果需要计算奖励）
        self.vocoder = None
        if self.reward_fn is not None:
            # 延迟加载vocoder，在train方法中初始化
            pass


    def _create_seed(self, prompts, base_seed):
        """为每个prompt创建确定性种子（如果same_latent=True）
        
        返回: list of seeds，如果 same_latent=False 则返回 None
        
        可复现性保证：
        1. 相同的 prompt 总是生成相同的 hash
        2. 相同的 base_seed + prompt_hash 总是生成相同的 seed
        3. 相同的 seed 总是生成相同的随机数序列（通过 torch.manual_seed(seed)）
        4. 因此，相同的 (prompt, epoch, batch_idx, sample_idx, k) 总是生成相同的初始噪声
        
        注意：只返回 seed 值，不创建 generator 对象，因为 sample() 方法使用 torch.manual_seed(seed)
        而不是直接使用 generator 对象。
        """
        if not self.same_latent:
            return None
        seeds = []
        for prompt in prompts:
            # 1. 计算 prompt 的哈希值（确定性：相同 prompt → 相同 hash）
            hash_digest = hashlib.sha256(str(prompt).encode()).digest()
            prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
            
            # 2. 结合 base_seed 生成最终种子（确定性：相同 base_seed + prompt_hash → 相同 seed）
            seed = (base_seed + prompt_hash_int) % (2**31)
            seeds.append(seed)
        return seeds
    
    def _batch_mel_to_audio(self, mel_list, batch_size=8):
        """批量将mel spectrogram转换为音频（高效版本）
        
        Args:
            mel_list: List of mel spectrograms, each with shape (gen_len, d)
            batch_size: Batch size for vocoder processing
        
        Returns:
            List of audio waveforms
        """
        if self.vocoder is None:
            return None
        
        audio_list = []
        
        # 如果mel数量较少，直接逐个处理
        if len(mel_list) <= batch_size:
            with torch.inference_mode():
                for mel in mel_list:
                    # 检查 mel 是否为空或形状不正确
                    if mel.numel() == 0 or mel.shape[0] == 0 or mel.shape[1] == 0:
                        print(f"⚠️  警告: 跳过空的 mel (shape={mel.shape})")
                        audio_list.append(torch.zeros(1))  # 添加占位符
                        continue
                    
                    # mel shape: (gen_len, d) -> (1, d, gen_len) for vocoder
                    mel_spec = mel.permute(1, 0).unsqueeze(0).to(torch.float32)  # (1, d, gen_len)
                    
                    # 确保 mel_spec 形状正确
                    if mel_spec.shape[1] == 0 or mel_spec.shape[2] == 0:
                        print(f"⚠️  警告: mel_spec 形状不正确 (shape={mel_spec.shape})，跳过")
                        audio_list.append(torch.zeros(1))  # 添加占位符
                        continue
                    
                    try:
                        if self.vocoder_name == "vocos":
                            audio = self.vocoder.decode(mel_spec).cpu()
                        elif self.vocoder_name == "bigvgan":
                            audio = self.vocoder(mel_spec).squeeze(0).cpu()
                        else:
                            raise ValueError(f"Unknown vocoder type: {self.vocoder_name}")
                        
                        audio_list.append(audio.squeeze())
                    except Exception as e:
                        print(f"⚠️  警告: vocoder 解码失败 (mel shape={mel.shape}, mel_spec shape={mel_spec.shape}): {e}")
                        audio_list.append(torch.zeros(1))  # 添加占位符
        else:
            # 批量处理：使用padding来批量处理
            with torch.inference_mode():
                # 过滤掉空的 mel
                valid_mels = [mel for mel in mel_list if mel.numel() > 0 and mel.shape[0] > 0 and mel.shape[1] > 0]
                if len(valid_mels) == 0:
                    print("⚠️  警告: 所有 mel 都为空，返回空列表")
                    return []
                
                # 找到最大长度
                max_len = max(mel.shape[0] for mel in valid_mels)
                
                # 分批处理
                for i in range(0, len(valid_mels), batch_size):
                    batch_mels = valid_mels[i:i+batch_size]
                    
                    # Padding到相同长度
                    batch_mel_specs = []
                    mel_lengths = []
                    for mel in batch_mels:
                        mel_len = mel.shape[0]
                        mel_lengths.append(mel_len)
                        # mel shape: (gen_len, d) -> (d, gen_len)
                        mel_spec = mel.permute(1, 0).to(torch.float32)  # (d, gen_len)
                        # Padding到max_len
                        if mel_len < max_len:
                            padding = torch.zeros(mel_spec.shape[0], max_len - mel_len, 
                                                dtype=mel_spec.dtype, device=mel_spec.device)
                            mel_spec = torch.cat([mel_spec, padding], dim=1)  # (d, max_len)
                        batch_mel_specs.append(mel_spec)
                    
                    # Stack成batch: (batch_size, d, max_len)
                    batch_mel_specs = torch.stack(batch_mel_specs).to(self.accelerator.device)
                    
                    # 批量解码
                    try:
                        if self.vocoder_name == "vocos":
                            batch_audio = self.vocoder.decode(batch_mel_specs).cpu()
                            # vocos 可能返回 (batch_size, audio_len) 或 (batch_size, 1, audio_len)
                            if batch_audio.ndim == 3:
                                # (batch_size, 1, audio_len)
                                batch_audio = batch_audio.squeeze(1)  # -> (batch_size, audio_len)
                        elif self.vocoder_name == "bigvgan":
                            batch_audio = self.vocoder(batch_mel_specs).cpu()  # (batch_size, audio_len)
                        else:
                            raise ValueError(f"Unknown vocoder type: {self.vocoder_name}")
                        
                        # 提取每个样本的音频（去掉padding部分）
                        for j, mel_len in enumerate(mel_lengths):
                            audio_len = int(mel_len * 256)  # hop_length = 256
                            # batch_audio 现在是 (batch_size, audio_len)
                            audio = batch_audio[j, :audio_len]
                            audio_list.append(audio)
                    except Exception as e:
                        print(f"⚠️  警告: 批量 vocoder 解码失败 (batch_mel_specs shape={batch_mel_specs.shape}): {e}")
                        # 为失败的 batch 添加占位符
                        for _ in batch_mels:
                            audio_list.append(torch.zeros(1))
        
        return audio_list
    
    def _compute_reward_mfa(self, emphasis_words, rank_id):
        print(f"mfa rank_id: {rank_id}")
        proc_dir = f"/data/F5-TTS/grpo_rewards/wavs_{rank_id}"
        output_dir = f"/data/F5-TTS/grpo_rewards/annotations_{rank_id}"
        json_output = f"/data/F5-TTS/grpo_rewards/prominence_{rank_id}.json"
        wav_out_dir = f"/data/F5-TTS/grpo_rewards/wavs_out_{rank_id}"
        for tgt_path in [output_dir, json_output, wav_out_dir]:
            if os.path.exists(tgt_path):
                if os.path.isfile(tgt_path):
                    os.remove(tgt_path)
                else:
                    shutil.rmtree(tgt_path)

        num_jobs = os.cpu_count()

        cmd = f"mfa align {proc_dir} mandarin_china_mfa mandarin_mfa {output_dir} --clean -j {num_jobs}"
        # 执行 MFA 对齐，捕获错误输出以便调试
        try:
            result = run(cmd, shell=True, check=True, stdout=PIPE, stderr=PIPE, text=True)
            # result = run(cmd, shell=True, check=True)
        except CalledProcessError as e:
            # 打印错误信息以便调试
            print(f"❌ MFA align 失败 (exit code {e.returncode}):")
            print(f"命令: {cmd}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            # 检查是否有音频文件
            if os.path.exists(proc_dir):
                wav_files = [f for f in os.listdir(proc_dir) if f.endswith('.wav')]
                print(f"音频文件数量: {len(wav_files)}")
                if len(wav_files) == 0:
                    print(f"⚠️  警告: {proc_dir} 目录中没有音频文件")
            # MFA 失败时由 _get_reward 读不到 JSON，会走零奖励分支，此处无需 return
    
    def _zero_rewards_list(self, num_samples, device):
        """返回与 emphasis_words 数量一致的零奖励 list，保证 torch.cat 后形状为 (num_samples,)。"""
        k = self.num_samples_per_prompt
        return [
            torch.zeros(min(k, num_samples - i), dtype=torch.float32, device=device)
            for i in range(0, num_samples, k)
        ]
    
    def _get_reward(self, emphasis_words, rank_id):
        proc_dir = f"/data/F5-TTS/grpo_rewards/wavs_{rank_id}"
        output_dir = f"/data/F5-TTS/grpo_rewards/annotations_{rank_id}"
        json_output = f"/data/F5-TTS/grpo_rewards/prominence_{rank_id}.json"
        wav_out_dir = f"/data/F5-TTS/grpo_rewards/wavs_out_{rank_id}"
        num_jobs = 20
        script_path = Path(__file__).resolve().parents[3] / "wavelet_prosody_toolkit" / "compute_prominence.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--wav_dir", str(proc_dir),
            "--annotation_dir", str(output_dir),
            "--json_output", str(json_output),
            "--wav_out_dir", str(wav_out_dir),
            "--num_workers", str(num_jobs),
        ]
        num_samples = len(emphasis_words)
        dev = self.accelerator.device
        try:
            result = run(cmd, check=True, stdout=PIPE, stderr=PIPE, text=True)
        except CalledProcessError as e:
            print(f"❌ compute_prominence 脚本执行失败 (exit code {e.returncode}):")
            print(f"命令: {' '.join(cmd)}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            return self._zero_rewards_list(num_samples, dev)

        if not os.path.exists(json_output):
            print(f"⚠️  警告: JSON 输出文件不存在: {json_output}")
            return self._zero_rewards_list(num_samples, dev)
        
        with open(json_output, "r") as f:
            data = json.load(f)
        
        rewards = []
        for idx, (_, item) in enumerate(data.items()):
            if idx >= num_samples:
                break
            emphasis_word = emphasis_words[idx]
            words = item.get("words", [])
            cnt = 0
            reward = 0.0
            for word in words:
                if ":" not in word:
                    continue
                w, v = word.split(":", 1)
                try:
                    v = float(v)
                except ValueError:
                    continue
                if w in emphasis_word or emphasis_word in w:
                    cnt += 1
                    reward = v
            if cnt == 1:
                rewards.append(reward)
            else:
                rewards.append(0.0)
        # 若 JSON 条目少于样本数，用 0 补齐
        while len(rewards) < num_samples:
            rewards.append(0.0)
        rewards = rewards[:num_samples]
        k = self.num_samples_per_prompt
        rewards_tensors = [
            torch.tensor(rewards[i : i + k], dtype=torch.float32, device=dev)
            for i in range(0, len(rewards), k)
        ]
        return rewards_tensors if rewards_tensors else self._zero_rewards_list(num_samples, dev)
                
    
    def train(
        self,
        train_dataset: Dataset,
        num_workers: int = 8,
        resumable_with_seed: int = None,
    ):
        # 初始化vocoder（如果需要计算奖励或记录样本）
        from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef
        
        self.vocoder = load_vocoder(
            vocoder_name=self.vocoder_name, 
            is_local=self.is_local_vocoder, 
            local_path=self.local_vocoder_path
        )
        # 将vocoder移到GPU（如果支持）
        if hasattr(self.vocoder, 'to'):
            self.vocoder = self.vocoder.to(self.accelerator.device)
        
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, nfe_step, sway_sampling_coef
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)
        else:
            from f5_tts.infer.utils_infer import sway_sampling_coef

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        # 指定 per-process batch_size，据此计算每 epoch 的 batch 数，使一 epoch 覆盖整个 dataset
        n_proc = self.accelerator.num_processes
        batch_size = max(1, self.sample_batch_size)  # 采样阶段每步的 prompt 数（per process）
        num_batches_per_epoch = max(1, math.ceil(len(train_dataset) / (batch_size * n_proc)))
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=grpo_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update

        # 保证 ref 与当前 model 一致：resume 时 load_checkpoint 只加载了 model，未更新 ref，
        # 会导致 ref≠model，第一次 sync 时 kl_loss 就不从 0 开始。grad_accumulation_steps>1 时，
        # 第一次 sync 前 32 次前向都未 step()，理论上 kl_loss 应为 0。
        ref_model = self.accelerator.unwrap_model(self.ref_model)
        ref_model.load_state_dict(
            self.accelerator.unwrap_model(self.model).state_dict(), strict=True
        )

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        # 创建训练迭代器
        train_iter = iter(train_dataloader)
        
        # 获取sway_sampling_coef、cfg_strength、nfe_step（采样步数须与 timesteps 一致）
        from f5_tts.infer.utils_infer import sway_sampling_coef, cfg_strength, nfe_step
        
        # 生成 timesteps：训练与采样共用；不再使用 odeint/EPSS，统一用均匀网格
        train_timesteps_full = torch.linspace(0, 1, nfe_step + 1, device=self.accelerator.device)
        if sway_sampling_coef is not None:
            train_timesteps_full = train_timesteps_full + sway_sampling_coef * (
                torch.cos(torch.pi / 2 * train_timesteps_full) - 1 + train_timesteps_full
            )
        
        epoch = skipped_epoch
        epoch = 0
        global_step = 0
        
        if self.accelerator.is_main_process:
            print(f"🚀 开始训练: epoch={epoch}/{self.epochs}, num_batches_per_epoch={num_batches_per_epoch}, batch_size={batch_size}, num_samples_per_prompt={self.num_samples_per_prompt}")
        
        while epoch < self.epochs:
            #################### SAMPLING ####################
            self.model.eval()
            samples = []
            all_prompts = []
            # 本 epoch 内按顺序递增的 uttid，保证文件夹按同名排序后与生成顺序一致
            sample_uttid_counter = 0
            

            for i in tqdm(
                range(num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not self.accelerator.is_local_main_process,
            ):
                # 获取batch数据
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)
                
                # 提取数据
                texts = batch["text"]
                text_lengths = batch["text_lengths"]
                raw_text = batch["raw_text"]
                emphasis_words = batch["emphasis_word"]
                emphasis_ids = batch.get("emphasis_ids", None)
                ref_mel = batch["ref_mel"].permute(0, 2, 1)  # (b, n, d)
                ref_mel_lengths = batch["ref_mel_lengths"]
                ref_text = batch["ref_text"]
                ref_emphasis_ids = batch.get("ref_emphasis_ids", None)
                ref_text_lengths = batch["ref_text_lengths"]
                local_speed = getattr(self, "local_speed", 1.0)

                # 为每个prompt采样k个样本
                batch_infer_texts = []
                batch_prompt_mels = []
                batch_raw_texts = []
                batch_emphasis_words = []
                batch_samples = []
                batch_log_probs = []
                batch_timesteps_list = []
                batch_latents_list = []
                batch_prev_latents_mean_list = []
                batch_emphasis_ids = []
                batch_durations = []
                batch_info = []

                for j in range(len(texts)):
                    ref_audio_len = ref_mel_lengths[j].item() if torch.is_tensor(ref_mel_lengths[j]) else int(ref_mel_lengths[j])
                    infer_text = [
                        ref_text[j] + [" "] + texts[j]
                    ]
                    infer_emphasis_ids = [ref_emphasis_ids[j] + [0] + emphasis_ids[j]]
                    ref_text_len = ref_text_lengths[j].item() if torch.is_tensor(ref_text_lengths[j]) else int(ref_text_lengths[j])
                    gen_text_len = text_lengths[j].item() if torch.is_tensor(text_lengths[j]) else int(text_lengths[j])
                    speed = float(local_speed) if torch.is_tensor(local_speed) else float(local_speed)
                    # 防止 ref_text_len 为 0 导致除零错误
                    if ref_text_len == 0:
                        print(f"⚠️  警告: ref_text_len=0，使用默认 duration")
                        duration = ref_audio_len * 2  # 使用与 trainer.py 相同的逻辑
                    else:
                        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

                    # 为这个prompt采样k个样本
                    # for k in range(self.num_samples_per_prompt):
                    seed_for_sample = None
                    if self.same_latent:
                        infer_text_str = " ".join(str(t) for t in infer_text)
                        seed_list = self._create_seed([infer_text_str], base_seed=epoch*10000+i*1000+j*10+self.num_samples_per_prompt)
                        seed_for_sample = seed_list[0]  # 直接使用种子值
                    
                    # 只取前 ref_audio_len 帧作为 cond，避免传入 padding（与 trainer.py 一致）
                    cond_mel = ref_mel[j:j+1, :ref_audio_len, :]  # (1, ref_audio_len, d)
                    # 确保 cond_mel 在正确的 device 和 dtype 上
                    cond_mel = cond_mel.to(self.accelerator.device)
                    
                    # 确保 infer_text 格式正确：应该是列表的列表，每个内部列表是一个 token 序列
                    # ref_text[j] 和 texts[j] 应该都是列表
                    if not isinstance(ref_text[j], list):
                        ref_text_tokens = list(ref_text[j]) if isinstance(ref_text[j], str) else [ref_text[j]]
                    else:
                        ref_text_tokens = ref_text[j]
                    if not isinstance(texts[j], list):
                        text_tokens = list(texts[j]) if isinstance(texts[j], str) else [texts[j]]
                    else:
                        text_tokens = texts[j]
                    infer_text = [ref_text_tokens + [" "] + text_tokens]
                    
                    cond_mel = cond_mel.repeat(self.num_samples_per_prompt, 1, 1)
                    infer_text = infer_text * self.num_samples_per_prompt
                    infer_emphasis_ids = infer_emphasis_ids * self.num_samples_per_prompt
                    duration = torch.tensor([duration] * self.num_samples_per_prompt, device=self.accelerator.device, dtype=torch.long)
                    
                    with torch.no_grad():  
                        generated, all_latents, all_log_probs, all_prev_latents_mean, timesteps_used = self.accelerator.unwrap_model(self.model).sample_with_logprob(
                            cond=cond_mel,
                            text=infer_text,
                            duration=duration,
                            steps=nfe_step,
                            cfg_strength=cfg_strength,
                            sway_sampling_coef=sway_sampling_coef,
                            emphasis_ids=infer_emphasis_ids,
                            seed=seed_for_sample,  # 传递种子值，sample() 内部会使用 torch.manual_seed(seed)
                            noise_level=self.noise_level,  # 传递 noise_level 参数
                        )
                    
                    generated = generated.to(torch.float32)
                    gen_mel_raw = generated[:, ref_audio_len:, :]  
                   
                    batch_info.append(
                        {"ref_text": ref_text[j], "text": texts[j], "ref_text_len": ref_text_len, "gen_text_len": gen_text_len}
                        for _ in range(self.num_samples_per_prompt)
                    )
                    batch_infer_texts.append(infer_text)
                    batch_prompt_mels.append(cond_mel)
                    batch_raw_texts.append(raw_text[j])
                    batch_emphasis_words.append(emphasis_words[j])
                    batch_emphasis_ids.append(infer_emphasis_ids)
                    batch_samples.append(gen_mel_raw)  # (k, gen_n, d)
                    batch_log_probs.append(all_log_probs)  # 已是 (k, num_steps)，cfm 内已 stack
                    batch_latents_list.append(torch.stack(all_latents))  # (k,num_steps, n, d)
                    batch_prev_latents_mean_list.append(torch.stack(all_prev_latents_mean))  # (k, num_steps, n, d)
                    batch_timesteps_list.append(timesteps_used)  # (nfe_step,) 由 sample_with_logprob 返回，与 log_probs 一致
                    batch_durations.append(duration)

                # 本 batch 内按 prompt 再按 sample 顺序：转成音频、起 uttid、保存，并拼成列表供 reward
                rank_id = self.accelerator.process_index
                wav_dir = f"/data/F5-TTS/grpo_rewards/wavs_{rank_id}"
                if os.path.exists(wav_dir):
                    shutil.rmtree(wav_dir)
                os.makedirs(wav_dir, exist_ok=True)

                batch_audio_samples = []
                for v in range(len(batch_samples)):
                    audios_v = self._batch_mel_to_audio(batch_samples[v])
                    raw = batch_raw_texts[v]
                    lab_text = raw if isinstance(raw, str) else " ".join(str(x) for x in raw)
                    lab_text = lab_text.replace("<strong>", "").replace("</strong>", "")
                    lab_text = lab_text.replace("[emphasis]", "").replace("[/emphasis]", "")
                    lab_text = re.sub(r"[，。！？；：、""''（）【】《》〈〉「」『』〖〗〘〙〚〛…—～·•]", "", lab_text)
                    lab_text = re.sub(r"[" + re.escape(string.punctuation) + r"]", "", lab_text)
                    lab_text = re.sub(r"\s+", " ", lab_text).strip()
                    for u in range(len(audios_v)):
                        uttid = f"{sample_uttid_counter:06d}"
                        torchaudio.save(f"{wav_dir}/{uttid}.wav", audios_v[u], target_sample_rate)
                        with open(f"{wav_dir}/{uttid}.lab", "w") as f:
                            f.write(lab_text)
                        batch_audio_samples.append(audios_v[u])
                        sample_uttid_counter += 1

                batch_emphasis_words_repeated = [
                    w for w in batch_emphasis_words
                    for _ in range(self.num_samples_per_prompt)
                ]

                self.accelerator.wait_for_everyone()
                if self.is_local_main_process:
                    for rank_id in range(self.accelerator.num_processes):
                        self._compute_reward_mfa(batch_emphasis_words_repeated, rank_id)

                self.accelerator.wait_for_everyone()
                rank_id = self.accelerator.process_index
                rewards_list = self._get_reward(batch_emphasis_words_repeated, rank_id)
                if not rewards_list:
                    rewards = torch.zeros(
                        len(batch_emphasis_words_repeated),
                        dtype=torch.float32,
                        device=self.accelerator.device,
                    )
                else:
                    rewards = torch.cat(rewards_list, dim=0)

                

                # batch_latents_list[i] 形状 (T+1, k, n, d)，第 1 维是 t，第 2 维是 k；对每个元素切片得到 step 前/后 latent
                samples.append({
                    "batch_info": batch_info,
                    "batch_infer_texts": batch_infer_texts,
                    "batch_emphasis_ids": batch_emphasis_ids,
                    "batch_prompt_mels": batch_prompt_mels,
                    "batch_log_probs": torch.stack(batch_log_probs),
                    "batch_latents": [x[:-1] for x in batch_latents_list],       # 每项 (T, k, n, d)
                    "batch_latents_next": [x[1:] for x in batch_latents_list],  # 每项 (T, k, n, d)
                    "batch_timesteps_list": torch.stack(batch_timesteps_list),
                    "batch_durations": batch_durations,
                    "rewards": rewards,
                })
            

            all_samples = defaultdict(list)
            for key in samples[0].keys():
                for item in samples:
                    val = item[key]
                    if key == "batch_infer_texts" or key == "batch_prompt_mels" \
                        or key == "batch_emphasis_ids" or key == "batch_durations":
                        for i in range(min(batch_size, len(val))):
                            for j in range(self.num_samples_per_prompt):
                                all_samples[key].append(val[i][j])
                    elif key == "batch_log_probs" or key == "batch_timesteps_list":
                        all_samples[key].append(val)
                    elif key == "batch_latents" or key == "batch_latents_next":
                        # val: list of (T, k, d, n)，d/n 可能不同，展平为 2*3 个 (32, d, n) 放入 list
                        for i in range(len(val)):
                            for j in range(val[i].shape[1]):
                                all_samples[key].append(val[i][:, j, :, :])  # (T, d, n)
                    elif key == "rewards":
                        all_samples[key].append(val)

            _log_probs = torch.cat(all_samples["batch_log_probs"], dim=0)  # (B, k, steps) 或 (2, 3, 32)
            all_samples["batch_log_probs"] = _log_probs.reshape(-1, _log_probs.shape[-1])  # (B*k, steps) 即 (6, 32)
            _timesteps = torch.cat(all_samples["batch_timesteps_list"], dim=0)
            all_samples["batch_timesteps_list"] = _timesteps.reshape(-1, _timesteps.shape[-1])
            all_samples["rewards"] = torch.cat(all_samples["rewards"], dim=0).float().unsqueeze(-1).repeat(1, all_samples["batch_timesteps_list"].shape[-1])
            
            # 诊断：记录每个 prompt 的 reward_std；per-prompt 无方差时用全局归一化，避免 advantage 全 0、policy_loss 归零
            min_prompt_std = 1e-6
            rewards_scalar_for_adv = all_samples["rewards"][:, 0]  # (N,)
            global_reward_mean = rewards_scalar_for_adv.float().mean()
            total_samples_actual = all_samples["rewards"].shape[0]  # 本 rank 实际样本数（分布式下可能少于 num_batches*bs*k）
            global_reward_std = rewards_scalar_for_adv.float().std()
            # if total_samples_actual > 1:
            #     global_reward_std = global_reward_std.clamp(min=1e-6)
            per_prompt_reward_stds = []
            all_samples["advantages"] = torch.zeros((total_samples_actual, 32), device=self.accelerator.device, dtype=torch.float32)
            num_zero_std_fallback = 0
            k_adv = self.num_samples_per_prompt
            num_prompts_adv = total_samples_actual // k_adv
            for u in range(num_prompts_adv):
                st, ed = u * k_adv, min((u + 1) * k_adv, total_samples_actual)
                rewards = all_samples["rewards"][st: ed].float()
                rewards_mean = rewards.mean()
                rewards_std = rewards.std()
                # if rewards.numel() > 1:
                #     rewards_std = rewards_std.clamp(min=1e-6)
                # per_prompt_reward_stds.append(rewards_std.item() if torch.is_tensor(rewards_std) else float(rewards_std))
                # if rewards_std < min_prompt_std:
                #     rewards_mean = global_reward_mean.float()
                #     rewards_std = global_reward_std.float()
                #     num_zero_std_fallback += 1
                advantages = (rewards - rewards_mean) / (rewards_std + 1e-4)
                all_samples["advantages"][st: ed] = advantages.float()
            if num_zero_std_fallback > 0 and self.accelerator.is_local_main_process:
                print(f"[GRPO] {num_zero_std_fallback}/{num_prompts_adv} prompts had zero reward_std → used global norm for advantage")

            # Epoch 级诊断：多少 prompt 的 reward 无方差（会导致该 prompt 的 advantage 全 0）
            num_prompts_total = num_prompts_adv
            num_zero_std_prompts = sum(1 for s in per_prompt_reward_stds if s < min_prompt_std)
            if num_prompts_total > 0 and len(per_prompt_reward_stds) > 0:
                per_prompt_stds_t = torch.tensor(per_prompt_reward_stds, device=self.accelerator.device)
                prompt_std_p25 = torch.quantile(per_prompt_stds_t.float(), 0.25).item()
                prompt_std_p50 = torch.quantile(per_prompt_stds_t.float(), 0.50).item()
                prompt_std_p75 = torch.quantile(per_prompt_stds_t.float(), 0.75).item()
                if self.accelerator.is_local_main_process:
                    print(f"[GRPO diagnostic] epoch {epoch}: prompts with reward_std<{min_prompt_std}: {num_zero_std_prompts}/{num_prompts_total} | reward_std p25/p50/p75: {prompt_std_p25:.6f} / {prompt_std_p50:.6f} / {prompt_std_p75:.6f}")

            # 采样 + advantage 阶段调试日志（便于追踪 reward/advantage 是否合理）
            with torch.no_grad():
                rewards_scalar = all_samples["rewards"][:, 0]  # (N,) 每样本一个标量
                adv_scalar = all_samples["advantages"][:, 0]     # (N,)
                log_probs_flat = all_samples["batch_log_probs"]  # (N, num_steps)
            total_samples = rewards_scalar.shape[0]
            reward_mean = rewards_scalar.float().mean().item()
            reward_std = rewards_scalar.float().std().item()
            reward_min = rewards_scalar.float().min().item()
            reward_max = rewards_scalar.float().max().item()
            adv_mean = adv_scalar.float().mean().item()
            adv_std = adv_scalar.float().std().item()
            log_prob_mean = log_probs_flat.float().mean().item()
            num_prompts = total_samples // self.num_samples_per_prompt
            if self.accelerator.is_main_process:
                self.writer.add_scalar("reward_mean", reward_mean, global_step)
                self.writer.add_scalar("reward_std", reward_std, global_step)
                self.writer.add_scalar("reward_min", reward_min, global_step)
                self.writer.add_scalar("reward_max", reward_max, global_step)
                self.writer.add_scalar("advantage_mean", adv_mean, global_step)
                self.writer.add_scalar("advantage_std", adv_std, global_step)
                self.writer.add_scalar("sample_log_prob_mean", log_prob_mean, global_step)
            rank_id = self.accelerator.process_index
            print(f"RANK: {rank_id}, [GRPO sample] reward mean={reward_mean:.4f} std={reward_std:.4f} range=[{reward_min:.4f},{reward_max:.4f}] | adv mean={adv_mean:.4f} std={adv_std:.4f} | log_prob mean={log_prob_mean:.4f} | N={total_samples} prompts={num_prompts}")

            #################### TRAINING ####################
            all_samples_size = len(all_samples["batch_infer_texts"])
            info = defaultdict(list)
            for inner_epoch in range(self.num_inner_epochs):
                self.model.train()
                info = defaultdict(list)
                
                train_batch_size = self.num_samples_per_prompt
                train_batch_size = 3
                for batch_idx in tqdm(
                    range(math.ceil(all_samples_size / train_batch_size)),
                    desc=f"Epoch {epoch}.{inner_epoch}: training",
                    disable=not self.accelerator.is_local_main_process,
                ):
                    start_idx = batch_idx * train_batch_size
                    end_idx = min(start_idx + train_batch_size, total_samples)
                    
                    # sample_info = all_samples["batch_info"][start_idx:end_idx]
                    sample_infer_text = all_samples["batch_infer_texts"][start_idx:end_idx]
                    sample_prompt_mel = all_samples["batch_prompt_mels"][start_idx:end_idx]
                    sample_ref_mel_lengths = torch.tensor([len(x) for x in sample_prompt_mel], device=self.accelerator.device, dtype=torch.long)
                    sample_advantages = all_samples["advantages"][start_idx:end_idx].detach().clone()
                    sample_emphasis_ids = all_samples["batch_emphasis_ids"][start_idx:end_idx]
                    sample_log_probs = all_samples["batch_log_probs"][start_idx:end_idx]
                    # [b, t, n, d]
                    sample_latents = all_samples["batch_latents"][start_idx:end_idx]
                    sample_latents_next = all_samples["batch_latents_next"][start_idx:end_idx]


                    # 诊断：当前 batch 的 advantage 统计（用于定位 policy_loss 归零）
                    with torch.no_grad():
                        batch_adv_flat = sample_advantages.float()
                        batch_adv_mean = batch_adv_flat.mean().item()
                        batch_adv_std = batch_adv_flat.std().item()
                        batch_adv_zero_frac = (batch_adv_flat.abs() < 1e-8).float().mean().item()
                    batch_diagnostic = {"batch_advantage_mean": batch_adv_mean, "batch_advantage_std": batch_adv_std, "batch_advantage_zero_frac": batch_adv_zero_frac}
                    total_batches = math.ceil(all_samples_size / train_batch_size)
                    if batch_adv_zero_frac >= 0.99 and self.accelerator.is_local_main_process:
                        print(f"[GRPO policy_loss≈0] step {global_step} (batch {batch_idx + 1}/{total_batches}, {100 * (batch_idx + 1) / total_batches:.0f}%): "
                              f"advantage_zero_frac={batch_adv_zero_frac:.2f} → 该 batch 的 advantage 全为 0，policy_loss 将归零")

                    # use samples with same prompt, length of mel is same
                    sample_latents = torch.stack(sample_latents, dim=0)
                    sample_latents_next = torch.stack(sample_latents_next, dim=0)

                    # mel padding
                    max_mel_length = sample_latents.shape[2]
                    padded_mels = []
                    for mel in sample_prompt_mel:
                        if mel.shape[0] < max_mel_length:
                            mel = F.pad(mel, (0, 0, 0, max_mel_length - mel.shape[0]), value=0)
                        padded_mels.append(mel)
                    sample_prompt_mel = torch.stack(padded_mels)

                    # span mask
                    b, t, _ = sample_prompt_mel.shape
                    span_mask = torch.ones((b, t), dtype=torch.bool, device=self.accelerator.device)


                    for j in tqdm(
                        range(len(train_timesteps_full) - 1),
                        desc="Timestep",
                        position=1,
                        leave=False,
                        disable=not self.accelerator.is_local_main_process,
                    ):
                        t_curr = train_timesteps_full[j]
                        t_next = train_timesteps_full[j + 1]
                        
                        with self.accelerator.accumulate(self.model):
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = self.model(
                                inp=sample_prompt_mel,
                                text=sample_infer_text,
                                lens=sample_ref_mel_lengths,
                                emphasis_ids=sample_emphasis_ids,
                                span_mask=span_mask,
                                is_grpo=True,
                                noise_level=self.noise_level,
                                t_curr=t_curr,
                                t_next=t_next,
                                latents=sample_latents[:, j],
                                latents_next=sample_latents_next[:, j],
                            )
                            if self.beta > 0:
                                with torch.no_grad():
                                    _, _, ref_prev_sample_mean, _ = self.ref_model(
                                        inp=sample_prompt_mel,
                                        text=sample_infer_text,
                                        lens=sample_ref_mel_lengths,
                                        emphasis_ids=sample_emphasis_ids,
                                        span_mask=span_mask,
                                        is_grpo=True,
                                        noise_level=self.noise_level,
                                        t_curr=t_curr,
                                        t_next=t_next,
                                        latents=sample_latents[:, j],
                                        latents_next=sample_latents_next[:, j],
                                    )
                       
                        advantages = torch.clip(
                            sample_advantages[:, j], 
                            -self.adv_clip_max, 
                            self.adv_clip_max
                        )
                        ratio = torch.exp(log_prob - sample_log_probs[:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio, 
                            1.0 - self.clip_range, 
                            1.0 + self.clip_range
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if self.beta > 0:
                            # 已关闭 dropout，policy 与 ref 前向一致，故仅在 j=0（本 step 尚未做任何 optimizer.step）时 KL≈0
                            std_dev_t_safe = std_dev_t.clamp(min=1e-2)
                            n_seq = prev_sample_mean.shape[1]
                            diff = (prev_sample_mean - ref_prev_sample_mean) ** 2  # (b, n, d)
                            gen_mask = (torch.arange(n_seq, device=prev_sample_mean.device).unsqueeze(0) >= sample_ref_mel_lengths.unsqueeze(1)).unsqueeze(-1)  # (b, n, 1)
                            diff_mask = gen_mask * diff
                            cnt_per_b = (gen_mask.expand_as(diff).sum(dim=(1, 2))).clamp(min=1)  # (b,)
                            diff_mean_per_b = diff_mask.sum(dim=(1, 2), keepdim=True) / cnt_per_b.unsqueeze(-1).unsqueeze(-1)  # (b, 1, 1)
                            kl_per_b = diff_mean_per_b.squeeze(-1).squeeze(-1) / (2 * std_dev_t_safe.squeeze(-1).squeeze(-1) ** 2)  # (b,)
                            kl_loss = kl_per_b.mean()
                            loss = policy_loss + self.beta * kl_loss
                        else:
                            loss = policy_loss
                        
                       
                        if j == 0:
                            for k, v in batch_diagnostic.items():
                                info[k].append(torch.tensor(v, device=self.accelerator.device))
                        info["approx_kl"].append(
                            0.5 
                            * torch.mean((log_prob - sample_log_probs[:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > self.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > self.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > self.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if self.beta > 0:
                            info["kl_loss"].append(kl_loss)
                            info["kl_loss_beta"].append(self.beta * kl_loss)

                        info["loss"].append(loss)
                        
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    if self.accelerator.sync_gradients:
                        # 对于 list 类型的值，使用 stack + mean；对于已经是 Tensor 的值，直接取 mean
                        processed_info = {}
                        for k, v in info.items():
                            # 先检查是否是 Tensor（优先级更高，因为某些键可能被直接赋值）
                            if isinstance(v, torch.Tensor):
                                processed_info[k] = torch.mean(v)
                            elif isinstance(v, list):
                                if len(v) == 0:
                                    # 空 list，跳过
                                    continue
                                elif len(v) == 1:
                                    # 单个元素，直接取 mean
                                    if isinstance(v[0], torch.Tensor):
                                        processed_info[k] = torch.mean(v[0])
                                    else:
                                        processed_info[k] = torch.tensor(v[0], dtype=torch.float32)
                                else:
                                    # 多个元素，使用 stack
                                    # 确保所有元素都是 Tensor
                                    tensor_list = []
                                    for item in v:
                                        if isinstance(item, torch.Tensor):
                                            tensor_list.append(item)
                                        else:
                                            tensor_list.append(torch.tensor(item, dtype=torch.float32))
                                    processed_info[k] = torch.mean(torch.stack(tensor_list))
                            else:
                                # 其他类型（如标量），尝试转换为 Tensor
                                try:
                                    processed_info[k] = torch.tensor(v, dtype=torch.float32)
                                except:
                                    processed_info[k] = v
                        info = processed_info
                        info = self.accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if self.is_main:
                            for k, v in info.items():
                                self.writer.add_scalar(f"{k}", v, global_step)
                            self.ema_model.update()
                        global_step += 1
                        info = defaultdict(list)
                    
                    if global_step % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                        print(f"Saving checkpoint at global_step {global_step}")
                        self.save_checkpoint(global_step)

            
            epoch += 1
        
        self.save_checkpoint(global_step, last=True)
        self.accelerator.end_training()
