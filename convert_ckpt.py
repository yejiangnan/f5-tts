import torch
from safetensors.torch import save_file
import os

def convert_pt_to_safetensors(pt_filename, sf_filename):
    print(f"正在加载 {pt_filename} ...")
    # 加载原始 checkpoint
    checkpoint = torch.load(pt_filename, map_location="cpu")
    
    weights = {}
    
    # --- F5-TTS 专用提取逻辑 ---
    # 优先寻找 EMA 权重 (通常用于推理，效果更好)
    if "ema_model_state_dict" in checkpoint:
        print("检测到 'ema_model_state_dict' (EMA权重)，正在提取用于推理...")
        weights = checkpoint["ema_model_state_dict"]
        
    # 如果没有 EMA，寻找普通权重
    elif "model_state_dict" in checkpoint:
        print("检测到 'model_state_dict'，正在提取...")
        weights = checkpoint["model_state_dict"]
        
    # 如果都没有，可能它本身就是权重字典
    else:
        print("未检测到嵌套结构，假设文件本身即为权重...")
        weights = checkpoint

    # --- 数据清理 ---
    # 1. 确保没有嵌套字典 (解决你的报错)
    # 检查 weights 的值是否包含非 Tensor 数据
    clean_weights = {}
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            # 处理多卡训练遗留的 'module.' 前缀 (F5-TTS 推理通常不需要这个前缀)
            new_key = k.replace("module.", "") 
            clean_weights[new_key] = v
        else:
            print(f"跳过非 Tensor 键: {k} (类型: {type(v)})")

    # 2. 检查显存连续性 (防止 share memory 报错)
    clean_weights = {k: v.contiguous() for k, v in clean_weights.items()}

    # --- 保存 ---
    print(f"正在保存为 {sf_filename} ...")
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(sf_filename), exist_ok=True)
    
    try:
        save_file(clean_weights, sf_filename)
        print(f"转换成功！文件已保存至: {sf_filename}")
    except Exception as e:
        print(f"转换最终失败: {e}")

if __name__ == "__main__":
    # 在这里修改你的路径
    input_pt = "/data/F5-TTS/ckpts/sft_data_emphasis_ids_enhanced_transformer/model_21500.pt"
    output_sf = "/data/F5-TTS/ckpts/F5TTS_v1_Base/model_21500.safetensors"
    
    convert_pt_to_safetensors(input_pt, output_sf)