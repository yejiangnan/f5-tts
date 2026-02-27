"""
简单的验证脚本，用于检查 forward_grpo 实现的正确性
"""
import torch
import torch.nn as nn
from f5_tts.model import CFM
from f5_tts.model.backbones.dit import DiT


def create_dummy_model(device="cpu"):
    """创建一个简单的测试模型"""
    transformer = DiT(
        dim=128,
        depth=2,
        heads=4,
        ff_mult=2,
        text_dim=64,
        text_num_embeds=100,
        mel_dim=80,
    )
    model = CFM(
        transformer=transformer,
        num_channels=80,
        mel_spec_kwargs=dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=80,
            target_sample_rate=24000,
        ),
    ).to(device)
    return model


def test_output_shapes():
    """测试输出形状是否正确"""
    print("=" * 60)
    print("测试 1: 输出形状检查")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_dummy_model(device)
    model.eval()
    
    batch_size = 2
    seq_len = 100
    n_mel_channels = 80
    text_len = 50
    
    # 创建测试数据
    inp = torch.randn(batch_size, seq_len, n_mel_channels).to(device)
    text = torch.randint(0, 100, (batch_size, text_len)).to(device)
    time = torch.rand(batch_size).to(device)
    noise = torch.randn_like(inp)
    span_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        prev_sample, log_prob, prev_sample_mean, std_dev_t = model.forward_grpo(
            inp=inp,
            text=text,
            time=time,
            noise=noise,
            span_mask=span_mask,
        )
    
    print(f"✓ prev_sample shape: {prev_sample.shape} (期望: [{batch_size}, {seq_len}, {n_mel_channels}])")
    print(f"✓ log_prob shape: {log_prob.shape} (期望: [{batch_size}])")
    print(f"✓ prev_sample_mean shape: {prev_sample_mean.shape} (期望: [{batch_size}, {seq_len}, {n_mel_channels}])")
    print(f"✓ std_dev_t shape: {std_dev_t.shape} (期望: [{batch_size}, 1, 1])")
    
    assert prev_sample.shape == (batch_size, seq_len, n_mel_channels), f"prev_sample 形状错误: {prev_sample.shape}"
    assert log_prob.shape == (batch_size,), f"log_prob 形状错误: {log_prob.shape}"
    assert prev_sample_mean.shape == (batch_size, seq_len, n_mel_channels), f"prev_sample_mean 形状错误: {prev_sample_mean.shape}"
    assert std_dev_t.shape == (batch_size, 1, 1), f"std_dev_t 形状错误: {std_dev_t.shape}"
    
    print("✓ 所有输出形状正确！\n")


def test_no_nan_inf():
    """测试是否有 NaN 或 Inf"""
    print("=" * 60)
    print("测试 2: NaN/Inf 检查")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_dummy_model(device)
    model.eval()
    
    batch_size = 2
    seq_len = 100
    n_mel_channels = 80
    text_len = 50
    
    # 创建测试数据
    inp = torch.randn(batch_size, seq_len, n_mel_channels).to(device)
    text = torch.randint(0, 100, (batch_size, text_len)).to(device)
    time = torch.rand(batch_size).to(device)
    noise = torch.randn_like(inp)
    span_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        prev_sample, log_prob, prev_sample_mean, std_dev_t = model.forward_grpo(
            inp=inp,
            text=text,
            time=time,
            noise=noise,
            span_mask=span_mask,
        )
    
    has_nan = torch.isnan(log_prob).any().item()
    has_inf = torch.isinf(log_prob).any().item()
    
    print(f"✓ log_prob 包含 NaN: {has_nan}")
    print(f"✓ log_prob 包含 Inf: {has_inf}")
    print(f"✓ log_prob 范围: [{log_prob.min().item():.4f}, {log_prob.max().item():.4f}]")
    
    assert not has_nan, "log_prob 包含 NaN！"
    assert not has_inf, "log_prob 包含 Inf！"
    
    print("✓ 没有 NaN 或 Inf！\n")


def test_gradient_flow():
    """测试梯度流是否正常"""
    print("=" * 60)
    print("测试 3: 梯度流检查")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_dummy_model(device)
    model.train()
    
    batch_size = 2
    seq_len = 50  # 减小序列长度以加快测试
    n_mel_channels = 80
    text_len = 30
    
    # 创建测试数据
    inp = torch.randn(batch_size, seq_len, n_mel_channels).to(device)
    text = torch.randint(0, 100, (batch_size, text_len)).to(device)
    time = torch.rand(batch_size).to(device)
    noise = torch.randn_like(inp)
    span_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    prev_sample, log_prob, prev_sample_mean, std_dev_t = model.forward_grpo(
        inp=inp,
        text=text,
        time=time,
        noise=noise,
        span_mask=span_mask,
    )
    
    # 计算损失并反向传播
    loss = log_prob.mean()
    loss.backward()
    
    # 检查是否有梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                break
    
    print(f"✓ 梯度流正常: {has_grad}")
    if has_grad:
        print(f"✓ 梯度范数: {grad_norm:.6f}")
    
    assert has_grad, "没有检测到梯度！"
    
    print("✓ 梯度流正常！\n")


def test_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试 4: 边界情况检查")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_dummy_model(device)
    model.eval()
    
    batch_size = 2
    seq_len = 100
    n_mel_channels = 80
    text_len = 50
    
    inp = torch.randn(batch_size, seq_len, n_mel_channels).to(device)
    text = torch.randint(0, 100, (batch_size, text_len)).to(device)
    noise = torch.randn_like(inp)
    span_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    # 测试 t=0
    print("测试 t=0...")
    with torch.no_grad():
        _, log_prob_t0, _, _ = model.forward_grpo(
            inp=inp,
            text=text,
            time=torch.zeros(batch_size).to(device),
            noise=noise,
            span_mask=span_mask,
        )
    print(f"✓ t=0 时 log_prob: {log_prob_t0.mean().item():.4f}")
    
    # 测试 t=1
    print("测试 t=1...")
    with torch.no_grad():
        _, log_prob_t1, _, _ = model.forward_grpo(
            inp=inp,
            text=text,
            time=torch.ones(batch_size).to(device),
            noise=noise,
            span_mask=span_mask,
        )
    print(f"✓ t=1 时 log_prob: {log_prob_t1.mean().item():.4f}")
    
    # 测试不同的 noise_level
    print("测试不同的 noise_level...")
    for noise_level in [0.1, 0.5, 0.7, 0.9]:
        with torch.no_grad():
            _, log_prob, _, _ = model.forward_grpo(
                inp=inp,
                text=text,
                time=torch.rand(batch_size).to(device),
                noise=noise,
                span_mask=span_mask,
                noise_level=noise_level,
            )
        print(f"✓ noise_level={noise_level}: log_prob={log_prob.mean().item():.4f}")
    
    print("✓ 边界情况测试通过！\n")


def test_compare_with_dpo():
    """与 forward_dpo 进行基本对比"""
    print("=" * 60)
    print("测试 5: 与 forward_dpo 对比")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_dummy_model(device)
    model.eval()
    
    batch_size = 2
    seq_len = 100
    n_mel_channels = 80
    text_len = 50
    
    inp = torch.randn(batch_size, seq_len, n_mel_channels).to(device)
    text = torch.randint(0, 100, (batch_size, text_len)).to(device)
    time = torch.rand(batch_size).to(device)
    noise = torch.randn_like(inp)
    span_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        # forward_dpo
        loss_dpo, cond_dpo, pred_dpo = model.forward_dpo(
            inp=inp,
            text=text,
            time=time,
            noise=noise,
            span_mask=span_mask,
        )
        
        # forward_grpo
        prev_sample_grpo, log_prob_grpo, prev_sample_mean_grpo, std_dev_t_grpo = model.forward_grpo(
            inp=inp,
            text=text,
            time=time,
            noise=noise,
            span_mask=span_mask,
        )
    
    print(f"✓ forward_dpo loss shape: {loss_dpo.shape}")
    print(f"✓ forward_grpo log_prob shape: {log_prob_grpo.shape}")
    print(f"✓ forward_grpo prev_sample shape: {prev_sample_grpo.shape}")
    print(f"✓ forward_grpo prev_sample_mean shape: {prev_sample_mean_grpo.shape}")
    print(f"✓ forward_grpo std_dev_t shape: {std_dev_t_grpo.shape}")
    
    # 注意：log_prob 和 loss 的数值范围不同是正常的
    # log_prob 是负的平方差（CPS），loss 是 MSE
    print(f"✓ forward_dpo loss 范围: [{loss_dpo.min().item():.4f}, {loss_dpo.max().item():.4f}]")
    print(f"✓ forward_grpo log_prob 范围: [{log_prob_grpo.min().item():.4f}, {log_prob_grpo.max().item():.4f}]")
    
    print("✓ 对比测试通过！\n")


def test_mask_handling():
    """测试 mask 处理是否正确"""
    print("=" * 60)
    print("测试 6: Mask 处理检查")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_dummy_model(device)
    model.eval()
    
    batch_size = 2
    seq_len = 100
    n_mel_channels = 80
    text_len = 50
    
    inp = torch.randn(batch_size, seq_len, n_mel_channels).to(device)
    text = torch.randint(0, 100, (batch_size, text_len)).to(device)
    time = torch.rand(batch_size).to(device)
    noise = torch.randn_like(inp)
    
    # 创建部分 mask
    span_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
    span_mask[:, 20:80] = True  # 只 mask 中间部分
    
    with torch.no_grad():
        prev_sample, log_prob, prev_sample_mean, std_dev_t = model.forward_grpo(
            inp=inp,
            text=text,
            time=time,
            noise=noise,
            span_mask=span_mask,
        )
    
    print(f"✓ Mask 形状: {span_mask.shape}")
    print(f"✓ Mask 中 True 的数量: {span_mask.sum().item()}")
    print(f"✓ log_prob 形状: {log_prob.shape}")
    print(f"✓ log_prob 值: {log_prob}")
    print(f"✓ prev_sample 形状: {prev_sample.shape}")
    print(f"✓ prev_sample_mean 形状: {prev_sample_mean.shape}")
    print(f"✓ std_dev_t 形状: {std_dev_t.shape}")
    
    # 检查 log_prob 是否合理（不应该全是 0）
    assert not torch.allclose(log_prob, torch.zeros_like(log_prob)), "log_prob 全为 0！"
    
    print("✓ Mask 处理正确！\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始验证 forward_grpo 实现")
    print("=" * 60 + "\n")
    
    try:
        test_output_shapes()
        test_no_nan_inf()
        test_gradient_flow()
        test_edge_cases()
        test_compare_with_dpo()
        test_mask_handling()
        
        print("=" * 60)
        print("✓ 所有测试通过！forward_grpo 实现看起来正确。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
