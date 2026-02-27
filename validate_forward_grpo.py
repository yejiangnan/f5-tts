"""
验证 forward_grpo 正确性的脚本
使用方法：在训练循环中调用这些验证函数
"""

import torch
import torch.nn.functional as F

def validate_forward_grpo_inputs(
    inp, text, latents, latents_next, lens, span_mask, 
    t_curr, t_next, batch_idx, timestep_idx
):
    """验证 forward_grpo 的输入"""
    print(f"\n=== 验证 forward_grpo 输入 (batch={batch_idx}, timestep={timestep_idx}) ===")
    
    # 1. 形状检查
    print(f"inp shape: {inp.shape}")
    print(f"latents shape: {latents.shape}")
    print(f"latents_next shape: {latents_next.shape}")
    print(f"text shape: {text.shape}")
    
    assert latents.shape[0] == inp.shape[0], f"batch mismatch: latents {latents.shape[0]} != inp {inp.shape[0]}"
    assert latents.shape[1] >= inp.shape[1], f"latents length {latents.shape[1]} < inp length {inp.shape[1]}"
    assert latents.shape == latents_next.shape, f"latents {latents.shape} != latents_next {latents_next.shape}"
    
    # 2. 数值范围检查
    print(f"inp range: [{inp.min().item():.4f}, {inp.max().item():.4f}]")
    print(f"latents range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
    print(f"t_curr: {t_curr}, t_next: {t_next}")
    
    assert not torch.isnan(inp).any(), "inp contains NaN"
    assert not torch.isnan(latents).any(), "latents contains NaN"
    assert 0 <= t_curr <= 1, f"t_curr {t_curr} not in [0, 1]"
    assert 0 <= t_next <= 1, f"t_next {t_next} not in [0, 1]"
    assert t_next > t_curr, f"t_next {t_next} <= t_curr {t_curr}"
    
    # 3. lens 检查
    if lens is not None:
        print(f"lens: {lens}")
        assert lens.max() <= inp.shape[1], f"lens max {lens.max()} > inp length {inp.shape[1]}"
    
    # 4. span_mask 检查
    if span_mask is not None:
        print(f"span_mask shape: {span_mask.shape}, True count: {span_mask.sum().item()}")
        assert span_mask.shape[0] == inp.shape[0], "span_mask batch mismatch"
    
    print("✓ 输入验证通过")


def validate_forward_grpo_outputs(
    prev_sample, log_prob, prev_sample_mean, std_dev_t,
    latents, latents_next, batch_idx, timestep_idx
):
    """验证 forward_grpo 的输出"""
    print(f"\n=== 验证 forward_grpo 输出 (batch={batch_idx}, timestep={timestep_idx}) ===")
    
    # 1. 形状检查
    print(f"prev_sample shape: {prev_sample.shape}")
    print(f"log_prob shape: {log_prob.shape}")
    print(f"prev_sample_mean shape: {prev_sample_mean.shape}")
    print(f"std_dev_t shape: {std_dev_t.shape}")
    
    assert prev_sample.shape == latents.shape, \
        f"prev_sample {prev_sample.shape} != latents {latents.shape}"
    assert log_prob.shape[0] == latents.shape[0], \
        f"log_prob batch {log_prob.shape[0]} != latents batch {latents.shape[0]}"
    
    # 2. 数值检查
    print(f"prev_sample range: [{prev_sample.min().item():.4f}, {prev_sample.max().item():.4f}]")
    print(f"log_prob range: [{log_prob.min().item():.4f}, {log_prob.max().item():.4f}]")
    print(f"prev_sample_mean range: [{prev_sample_mean.min().item():.4f}, {prev_sample_mean.max().item():.4f}]")
    print(f"std_dev_t: {std_dev_t.item():.6f}")
    
    assert not torch.isnan(prev_sample).any(), "prev_sample contains NaN"
    assert not torch.isnan(log_prob).any(), "log_prob contains NaN"
    assert not torch.isinf(prev_sample).any(), "prev_sample contains Inf"
    assert not torch.isinf(log_prob).any(), "log_prob contains Inf"
    assert std_dev_t > 0, f"std_dev_t {std_dev_t} <= 0"
    
    # 3. 与 latents_next 的对比（如果提供）
    if latents_next is not None:
        diff = (prev_sample - latents_next).abs().mean().item()
        print(f"prev_sample vs latents_next mean diff: {diff:.6f}")
        # 注意：diff 不应该为 0（因为 prev_sample 是预测值，latents_next 是真实值）
    
    print("✓ 输出验证通过")


def compare_with_inference(
    model, inp, text, latents, t_curr, t_next, 
    lens, emphasis_ids, span_mask, noise_level
):
    """对比训练时的 forward_grpo 和推理时的行为"""
    print("\n=== 对比训练 vs 推理 ===")
    
    # 训练模式
    model.train()
    with torch.enable_grad():
        train_outputs = model(
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
            latents_next=None,  # 推理时可能没有
        )
    
    # 推理模式
    model.eval()
    with torch.no_grad():
        eval_outputs = model(
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
            latents_next=None,
        )
    
    # 对比输出
    for name, train_out, eval_out in zip(
        ["prev_sample", "log_prob", "prev_sample_mean"],
        train_outputs[:3],
        eval_outputs[:3]
    ):
        if train_out is not None and eval_out is not None:
            diff = (train_out - eval_out).abs().mean().item()
            print(f"{name} train vs eval diff: {diff:.6f}")
    
    print("✓ 训练/推理对比完成")


def check_gradient_flow(model, loss):
    """检查梯度流"""
    print("\n=== 检查梯度流 ===")
    
    has_grad = False
    no_grad = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                has_grad = True
                if grad_norm < 1e-7:
                    no_grad.append(name)
        else:
            no_grad.append(name)
    
    if has_grad:
        print(f"✓ 梯度正常，{len(no_grad)} 个参数无梯度")
    else:
        print(f"⚠ 警告：没有检测到有效梯度")
    
    if len(no_grad) > 0 and len(no_grad) < 10:
        print(f"无梯度的参数: {no_grad[:5]}...")


# 使用示例：
# 在 trainer_grpo.py 的训练循环中：
# 
# from validate_forward_grpo import validate_forward_grpo_inputs, validate_forward_grpo_outputs
# 
# # 在调用 model 之前
# if j == 0 and batch_idx == 0:  # 只验证第一个 batch 的第一个 timestep
#     validate_forward_grpo_inputs(
#         sample_prompt_mel, sample_infer_text, 
#         sample_latents[:, j], sample_latents_next[:, j],
#         sample_ref_mel_lengths, span_mask, t_curr, t_next,
#         batch_idx, j
#     )
# 
# # 在调用 model 之后
# if j == 0 and batch_idx == 0:
#     validate_forward_grpo_outputs(
#         prev_sample, log_prob, prev_sample_mean, std_dev_t,
#         sample_latents[:, j], sample_latents_next[:, j],
#         batch_idx, j
#     )
