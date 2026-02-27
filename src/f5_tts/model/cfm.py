"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import math
from random import random
from typing import Callable

from pydantic import condecimal
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device
    
    
    def _step_with_logprob(
        self,
        pred,
        batch_size,
        t_curr,
        t_next,
        sample,
        noise_level,
        prev_sample = None,
        generator = None,
        return_sqrt_dt = None,
        cond_lens: torch.Tensor | None = None,
    ):
        # log_prob：仅对「生成段」求平均，排除 ref（cond）长度，否则会把 conditioning 的似然也算进 policy gradient
        pred = pred.float()
        t_curr_expanded = t_curr.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, 1).float()
        t_next_expanded = t_next.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, 1).float()

        sigma = t_curr_expanded
        sigma_prev = t_next_expanded
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)

        pred_original_sample = sample - sigma * pred
        noise_estimate = sample + pred * (1 - sigma)
        sqrt_term = torch.sqrt(sigma_prev ** 2 - std_dev_t ** 2)
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * sqrt_term

        if prev_sample is None:
            variance_noise = torch.randn(
                pred.shape,
                generator=generator,
                device=pred.device,
                dtype=pred.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)  # (b, n, d)

        if cond_lens is not None and log_prob.ndim == 3:
            seq_len = log_prob.shape[1]
            gen_mask = torch.arange(seq_len, device=log_prob.device).unsqueeze(0) >= cond_lens.unsqueeze(1)  # (b, n)
            log_prob_2d = log_prob.mean(dim=-1)
            cnt = gen_mask.sum(dim=1).clamp(min=1)
            log_prob_per_sample = (log_prob_2d * gen_mask).sum(dim=1) / cnt
        else:
            log_prob_per_sample = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return prev_sample, log_prob_per_sample, prev_sample_mean, std_dev_t
            
        
    

    def sample_with_logprob(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        emphasis_ids: int["b nt"] | None = None,
        noise_level: float = 0.1,  # CPS noise level parameter
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels
        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if isinstance(emphasis_ids, list):
            # Handle both single sample [1, 0, 1] and batch [[1, 0, 1], [0, 1, 0]]
            if len(emphasis_ids) > 0 and isinstance(emphasis_ids[0], list):
                # Multiple samples: use pad_sequence
                emphasis_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in emphasis_ids]
                emphasis_ids = pad_sequence(emphasis_ids_list, padding_value=0, batch_first=True).to(device)
            else:
                # Single sample: convert to tensor and add batch dimension
                emphasis_ids = torch.tensor(emphasis_ids, dtype=torch.long).to(device)
                emphasis_ids = emphasis_ids.unsqueeze(0)
        
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)


        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # noise input
        # 若提供 seed：每个样本用 seed+k 得到不同初始噪声，既有多样性又可复现（同 (seed, k) 得同噪声）
        y0 = []
        for k, dur in enumerate(duration):
            if exists(seed):
                torch.manual_seed(seed + k)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        
        # ✅ 手动实现去噪循环（替换 odeint），同时计算 log_prob
        # 与 step_cond 同 dtype，避免推理时 fp16 模型收到 fp32 的 time 导致 mat1/mat2 dtype 不一致
        dtype = step_cond.dtype
        y = y0.to(dtype)
        t = t.to(dtype)

        all_latents = [y0]
        all_log_probs = []
        all_prev_latents_mean = []
        for i in range(len(t) - 1):
            t_curr = t[i]  # 当前时间步（scalar tensor），第一次循环时通常是 t[0]=0
            t_next = t[i + 1]  # 下一个时间步（scalar tensor）
            
            # 计算 flow prediction（和 fn(t, x) 一样）
            # 注意：transformer 的 time 参数可以是 scalar 或 tensor，内部会处理

            

            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=y,
                    cond=step_cond,
                    text=text,
                    time=t_curr,  # scalar tensor，transformer 会扩展到 batch
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                    emphasis_ids=emphasis_ids,
                )
            else:
                # CFG: transformer 内部会处理复制（通过 cfg_infer=True）
                pred_cfg = self.transformer(
                    x=y,
                    cond=step_cond,
                    text=text,
                    time=t_curr,  # scalar tensor
                    mask=mask,
                    cfg_infer=True,
                    cache=True,
                    emphasis_ids=emphasis_ids,
                )
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                pred = pred + (pred - null_pred) * cfg_strength
            
            prev_sample, log_prob_per_sample, prev_sample_mean, std_dev_t \
                = self._step_with_logprob(
                    pred=pred,
                    batch_size=batch,
                    t_curr=t_curr,
                    t_next=t_next,
                    sample=y.float(),
                    noise_level=noise_level,
                    cond_lens=lens,
                )

            # 保持 y 与模型同 dtype，避免下一轮 transformer 出现 Float vs Half
            y = prev_sample.to(dtype)

            all_latents.append(y)
            all_log_probs.append(log_prob_per_sample)
            all_prev_latents_mean.append(prev_sample_mean)
        
        self.transformer.clear_cache()
        
        # 堆叠 log_probs
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps)
        
        sampled = all_latents[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        # 返回实际使用的 timesteps，每样本一份，形状 (batch_size, steps) 与 all_log_probs 对齐
        timesteps_used = t[:-1].unsqueeze(0).expand(batch, -1)  # (batch_size, steps)

        return out, all_latents, all_log_probs, all_prev_latents_mean, timesteps_used


    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        emphasis_ids: int["b nt"] | None = None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels
        print(f"cond.shape: {cond.shape}")
        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if isinstance(emphasis_ids, list):
            # Handle both single sample [1, 0, 1] and batch [[1, 0, 1], [0, 1, 0]]
            if len(emphasis_ids) > 0 and isinstance(emphasis_ids[0], list):
                # Multiple samples: use pad_sequence
                emphasis_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in emphasis_ids]
                emphasis_ids = pad_sequence(emphasis_ids_list, padding_value=0, batch_first=True).to(device)
            else:
                # Single sample: convert to tensor and add batch dimension
                emphasis_ids = torch.tensor(emphasis_ids, dtype=torch.long).to(device)
                emphasis_ids = emphasis_ids.unsqueeze(0)

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                    emphasis_ids=emphasis_ids,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
                emphasis_ids=emphasis_ids,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        time: float["b"] | float[""] = None,
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
        emphasis_ids: int["b nt"] | None = None,

        # dpo
        is_dpo: bool = False,
        noise: float["b n d"] | None = None,
        span_mask: bool["b n"] | None = None,  

        # grpo
        is_grpo: bool = False,
        noise_level: float = 0.2,  # CPS noise level parameter
        t_curr: float["b"] | float[""] = None,
        t_next: float["b"] | float[""] = None,
        latents: float["b n d"] | None = None,
        latents_next: float["b n d"] | None = None,
    ):
        if is_dpo:
            return self.forward_dpo(
                inp, 
                text, 
                time, 
                lens=lens, 
                noise_scheduler=noise_scheduler, 
                emphasis_ids=emphasis_ids, 
                noise=noise, 
                span_mask=span_mask,
            )

        if is_grpo:
            return self.forward_grpo(
                inp, 
                text, 
                lens=lens, 
                emphasis_ids=emphasis_ids,
                span_mask=span_mask,
                noise_level=noise_level,
                t_curr=t_curr,
                t_next=t_next,
                latents=latents,
                latents_next=latents_next,
            )

        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
            
        # handle emphasis_ids like text (as list)
        if isinstance(emphasis_ids, list):
            # Handle both single sample [1, 0, 1] and batch [[1, 0, 1], [0, 1, 0]]
            if len(emphasis_ids) > 0 and isinstance(emphasis_ids[0], list):
                # Multiple samples: use pad_sequence
                emphasis_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in emphasis_ids]
                emphasis_ids = pad_sequence(emphasis_ids_list, padding_value=0, batch_first=True).to(device)
            else:
                # Single sample: convert to tensor and add batch dimension
                emphasis_ids = torch.tensor(emphasis_ids, dtype=torch.long).to(device)
                emphasis_ids = emphasis_ids.unsqueeze(0)

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        if not exists(time):
            time = torch.rand((batch,), dtype=dtype, device=self.device)

        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask, emphasis_ids=emphasis_ids
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred

    def forward_dpo(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        time: float["b"] | float[""],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
        emphasis_ids: int["b nt"] | None = None,

        noise: float["b n d"] | None = None,
        span_mask: bool["b n"] | None = None,  
    ):

        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
            
        # handle emphasis_ids like text (as list)
        if isinstance(emphasis_ids, list):
            # Handle both single sample [1, 0, 1] and batch [[1, 0, 1], [0, 1, 0]]
            if len(emphasis_ids) > 0 and isinstance(emphasis_ids[0], list):
                # Multiple samples: use pad_sequence
                emphasis_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in emphasis_ids]
                emphasis_ids = pad_sequence(emphasis_ids_list, padding_value=0, batch_first=True).to(device)
            else:
                # Single sample: convert to tensor and add batch dimension
                emphasis_ids = torch.tensor(emphasis_ids, dtype=torch.long).to(device)
                emphasis_ids = emphasis_ids.unsqueeze(0)

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)
        rand_span_mask = span_mask
        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = noise


        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = False
        drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask, emphasis_ids=emphasis_ids
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        if rand_span_mask.ndim == 2:
            mask_expanded = rand_span_mask.unsqueeze(-1) # [B, T, 1]
        else:
            mask_expanded = rand_span_mask

        loss = loss * mask_expanded.type_as(loss)
        loss_sum = loss.sum(dim=[1, 2])
        num_elements = mask_expanded.sum(dim=1).squeeze(-1) * self.num_channels
        
        loss_per_sample = loss_sum / (num_elements + 1e-8)

        # 返回 [Batch_Size] 大小的向量，而不是标量
        return loss_per_sample, cond, pred


    def forward_grpo(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        cfg_strength: float = 1.0,
        emphasis_ids: int["b nt"] | None = None,
        noise: float["b n d"] | None = None,
        span_mask: bool["b n"] | None = None,
        noise_level: float = 0.7,  # CPS noise level parameter
        t_curr: float["b"] | float[""] = None,
        t_next: float["b"] | float[""] = None,
        latents: float["b n d"] | None = None,
        latents_next: float["b n d"] | None = None,
    ):
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
            
        # handle emphasis_ids like text (as list)
        if isinstance(emphasis_ids, list):
            # Handle both single sample [1, 0, 1] and batch [[1, 0, 1], [0, 1, 0]]
            if len(emphasis_ids) > 0 and isinstance(emphasis_ids[0], list):
                # Multiple samples: use pad_sequence
                emphasis_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in emphasis_ids]
                emphasis_ids = pad_sequence(emphasis_ids_list, padding_value=0, batch_first=True).to(device)
            else:
                # Single sample: convert to tensor and add batch dimension
                emphasis_ids = torch.tensor(emphasis_ids, dtype=torch.long).to(device)
                emphasis_ids = emphasis_ids.unsqueeze(0)
        
        # 创建 cond：基于 inp（prompt mel）
       
        cond = inp
        drop_audio_cond = False
        drop_text = False

        if cfg_strength < 1e-5:
            pred = self.transformer(
                x=latents,
                cond=cond,
                text=text,
                time=t_curr,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                emphasis_ids=emphasis_ids,
            )
        else:
            pred_cfg = self.transformer(
                x=latents,
                cond=cond,
                text=text,
                time=t_curr,
                cfg_infer=True,
                emphasis_ids=emphasis_ids,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            pred = pred + (pred - null_pred) * cfg_strength

        prev_sample, log_prob_per_sample, prev_sample_mean, std_dev_t \
            = self._step_with_logprob(
                pred=pred,
                batch_size=batch,
                t_curr=t_curr,
                t_next=t_next,
                sample=latents,
                prev_sample=latents_next,
                noise_level=noise_level,
                cond_lens=lens,
            )

        return prev_sample, log_prob_per_sample, prev_sample_mean, std_dev_t