"""
验证 forward_grpo 数值正确性：对比与 sample_with_logprob 的输出是否一致
"""

import torch
import torch.nn.functional as F

def validate_numerical_consistency(
    model,
    inp,  # prompt mel
    text,
    latents,  # 当前时间步的 latent
    latents_next,  # 下一个时间步的 latent（ground truth）
    t_curr,
    t_next,
    lens,
    emphasis_ids,
    span_mask,
    noise_level,
    seed=42,
    tolerance=1e-4,
    train_timesteps_full=None,  # 训练时使用的时间步序列（可选）
):
    """
    验证 forward_grpo 的输出是否与 sample_with_logprob 中对应时间步的输出一致
    
    方法：
    1. 用 forward_grpo 计算 prev_sample
    2. 用 sample_with_logprob 的完整轨迹，提取对应时间步的 prev_sample
    3. 对比两者是否一致（在固定随机种子的情况下）
    """
    print("\n" + "="*60)
    print("验证 forward_grpo 数值正确性")
    print("="*60)
    
    device = inp.device
    batch_size = inp.shape[0]
    
    # 1. 使用 forward_grpo 计算
    print("\n[1] 使用 forward_grpo 计算...")
    model.eval()
    with torch.no_grad():
        # 固定随机种子
        torch.manual_seed(seed)
        prev_sample_grpo, log_prob_grpo, prev_sample_mean_grpo, std_dev_t_grpo = model(
            inp=inp,
            text=text,
            lens=lens,
            emphasis_ids=emphasis_ids,
            span_mask=span_mask,
            is_grpo=True,
            noise_level=noise_level,
            t_curr=t_curr,
            t_next=t_next,
            latents=latents,
            latents_next=latents_next,
        )
    
    print(f"  prev_sample_grpo shape: {prev_sample_grpo.shape}")
    print(f"  prev_sample_mean_grpo range: [{prev_sample_mean_grpo.min().item():.6f}, {prev_sample_mean_grpo.max().item():.6f}]")
    print(f"  std_dev_t_grpo: {std_dev_t_grpo.item():.6f}")
    print(f"  log_prob_grpo range: [{log_prob_grpo.min().item():.6f}, {log_prob_grpo.max().item():.6f}]")
    
    # 2. 使用 sample_with_logprob 的完整轨迹，找到对应时间步
    print("\n[2] 使用 sample_with_logprob 计算完整轨迹...")
    
    # 计算 duration（latents 的长度）
    duration = latents.shape[1]
    duration_tensor = torch.tensor([duration] * batch_size, device=device, dtype=torch.long)
    
    # 使用与训练时相同的时间步序列
    if train_timesteps_full is not None:
        # 使用训练时的时间步序列
        t_full = train_timesteps_full
        steps = len(t_full) - 1
        print(f"  使用训练时的时间步序列，steps={steps}")
    else:
        # 使用默认时间步序列
        steps = 32
        t_full = torch.linspace(0, 1, steps + 1, device=device)
        print(f"  使用默认时间步序列，steps={steps}")
    
    # 找到最接近 t_curr 的时间步索引（all_latents[i] 对应从 t[i] 到 t[i+1] 的 prev_sample）
    t_curr_idx = torch.argmin(torch.abs(t_full[:-1] - t_curr))  # 注意：all_latents 长度是 steps，对应 t[:-1]
    t_next_in_full = t_full[t_curr_idx + 1] if t_curr_idx + 1 < len(t_full) else t_full[-1]
    
    print(f"  t_curr={t_curr.item():.6f}, 对应索引={t_curr_idx.item()}, t_full[{t_curr_idx}]={t_full[t_curr_idx].item():.6f}")
    print(f"  t_next={t_next.item():.6f}, t_full[{t_curr_idx+1}]={t_next_in_full.item():.6f}")
    t_curr_diff = abs(t_curr.item() - t_full[t_curr_idx].item())
    t_next_diff = abs(t_next.item() - t_next_in_full.item())
    print(f"  时间步差异: t_curr diff={t_curr_diff:.8f}, t_next diff={t_next_diff:.8f}")
    
    # 如果时间步差异太大，警告
    if t_curr_diff > 0.01 or t_next_diff > 0.01:
        print(f"  ⚠ 警告：时间步差异较大，可能影响验证准确性")
    
    # 使用 sample_with_logprob 计算完整轨迹
    # 需要传入与训练时相同的参数（sway_sampling_coef 等）
    torch.manual_seed(seed)
    
    # 获取 sway_sampling_coef（如果可用）
    try:
        from f5_tts.infer.utils_infer import sway_sampling_coef
    except:
        sway_sampling_coef = None
    
    _, all_latents_sample, all_log_probs_sample, all_prev_latents_mean_sample, timesteps_used = model.sample_with_logprob(
        cond=inp,
        text=text,
        duration=duration_tensor,
        lens=lens,
        steps=steps,
        cfg_strength=1.0,
        sway_sampling_coef=sway_sampling_coef,  # 使用与训练时相同的 sway_sampling_coef
        emphasis_ids=emphasis_ids,
        seed=seed,
        noise_level=noise_level,
    )
    
    # all_latents_sample 是列表，每个元素是一个时间步的 latent
    # all_latents_sample[i] 对应从 t[i] 到 t[i+1] 的 prev_sample
    # 所以 all_latents_sample[t_curr_idx] 应该对应从 t_full[t_curr_idx] 到 t_full[t_curr_idx+1] 的 prev_sample
    # 如果 t_curr 和 t_full[t_curr_idx] 很接近，那么结果应该一致
    
    if t_curr_idx < len(all_latents_sample) and abs(t_curr.item() - t_full[t_curr_idx].item()) < 0.01:
        prev_sample_sample = all_latents_sample[t_curr_idx]
        prev_sample_mean_sample = all_prev_latents_mean_sample[t_curr_idx] if t_curr_idx < len(all_prev_latents_mean_sample) else None
        log_prob_sample = all_log_probs_sample[:, t_curr_idx] if t_curr_idx < all_log_probs_sample.shape[1] else None
        
        print(f"\n[3] 对比结果:")
        print(f"  prev_sample_sample shape: {prev_sample_sample.shape}")
        
        # 对比 prev_sample_mean（不涉及随机采样，更可靠）
        if prev_sample_mean_sample is not None:
            mean_diff = (prev_sample_mean_grpo - prev_sample_mean_sample).abs().mean().item()
            max_diff = (prev_sample_mean_grpo - prev_sample_mean_sample).abs().max().item()
            print(f"\n  prev_sample_mean 对比:")
            print(f"    mean diff: {mean_diff:.8f}")
            print(f"    max diff: {max_diff:.8f}")
            
            if mean_diff < tolerance and max_diff < tolerance:
                print(f"    ✓ prev_sample_mean 一致（tolerance={tolerance}）")
            else:
                print(f"    ✗ prev_sample_mean 不一致！")
                print(f"      prev_sample_mean_grpo[0, 0, :5]: {prev_sample_mean_grpo[0, 0, :5]}")
                print(f"      prev_sample_mean_sample[0, 0, :5]: {prev_sample_mean_sample[0, 0, :5]}")
        
        # 对比 prev_sample（涉及随机采样，需要相同种子）
        sample_diff = (prev_sample_grpo - prev_sample_sample).abs().mean().item()
        sample_max_diff = (prev_sample_grpo - prev_sample_sample).abs().max().item()
        print(f"\n  prev_sample 对比:")
        print(f"    mean diff: {sample_diff:.8f}")
        print(f"    max diff: {sample_max_diff:.8f}")
        
        if sample_diff < tolerance and sample_max_diff < tolerance:
            print(f"    ✓ prev_sample 一致（tolerance={tolerance}）")
        else:
            print(f"    ⚠ prev_sample 有差异（可能是随机采样导致的）")
            print(f"      如果差异较小（<0.1），可能是正常的随机性")
            print(f"      prev_sample_grpo[0, 0, :5]: {prev_sample_grpo[0, 0, :5]}")
            print(f"      prev_sample_sample[0, 0, :5]: {prev_sample_sample[0, 0, :5]}")
        
        # 对比 log_prob
        if log_prob_sample is not None:
            log_prob_diff = (log_prob_grpo - log_prob_sample).abs().mean().item()
            log_prob_max_diff = (log_prob_grpo - log_prob_sample).abs().max().item()
            print(f"\n  log_prob 对比:")
            print(f"    mean diff: {log_prob_diff:.8f}")
            print(f"    max diff: {log_prob_max_diff:.8f}")
            
            if log_prob_diff < tolerance and log_prob_max_diff < tolerance:
                print(f"    ✓ log_prob 一致（tolerance={tolerance}）")
            else:
                print(f"    ✗ log_prob 不一致！")
                print(f"      log_prob_grpo: {log_prob_grpo}")
                print(f"      log_prob_sample: {log_prob_sample}")
        
        # 对比 latents_next（ground truth）
        if latents_next is not None:
            next_diff = (prev_sample_grpo - latents_next).abs().mean().item()
            print(f"\n  prev_sample vs latents_next (ground truth):")
            print(f"    mean diff: {next_diff:.8f}")
            print(f"    注意：这是预测值与真实值的差异，不应该为 0")
        
        print("\n" + "="*60)
        
        # 返回是否一致
        is_consistent = (
            (prev_sample_mean_sample is None or (mean_diff < tolerance and max_diff < tolerance)) and
            (log_prob_sample is None or (log_prob_diff < tolerance and log_prob_max_diff < tolerance))
        )
        
        return is_consistent, {
            'mean_diff': mean_diff if prev_sample_mean_sample is not None else None,
            'sample_diff': sample_diff,
            'log_prob_diff': log_prob_diff if log_prob_sample is not None else None,
        }
    else:
        print(f"  ⚠ 警告：t_curr_idx {t_curr_idx} >= len(all_latents_sample) {len(all_latents_sample)}")
        return False, {}


def validate_step_by_step(
    model,
    inp,
    text,
    latents_list,  # 所有时间步的 latents（从 sample_with_logprob 得到）
    t_list,  # 所有时间步
    lens,
    emphasis_ids,
    span_mask,
    noise_level,
    seed=42,
):
    """
    逐步验证：对每个时间步，用 forward_grpo 计算，对比与 sample_with_logprob 的结果
    """
    print("\n" + "="*60)
    print("逐步验证 forward_grpo")
    print("="*60)
    
    all_consistent = True
    for i in range(len(t_list) - 1):
        t_curr = t_list[i]
        t_next = t_list[i + 1]
        latents = latents_list[i]
        latents_next = latents_list[i + 1] if i + 1 < len(latents_list) else None
        
        print(f"\n时间步 {i}: t_curr={t_curr.item():.6f}, t_next={t_next.item():.6f}")
        
        is_consistent, diffs = validate_numerical_consistency(
            model, inp, text, latents, latents_next,
            t_curr, t_next, lens, emphasis_ids, span_mask, noise_level, seed
        )
        
        if not is_consistent:
            all_consistent = False
            print(f"  ✗ 时间步 {i} 不一致")
        else:
            print(f"  ✓ 时间步 {i} 一致")
    
    print("\n" + "="*60)
    if all_consistent:
        print("✓ 所有时间步验证通过！forward_grpo 数值正确")
    else:
        print("✗ 部分时间步验证失败，请检查实现")
    
    return all_consistent
