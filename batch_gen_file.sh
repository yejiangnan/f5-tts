#!/bin/bash

# 批量生成脚本：从 gen_file 读取文本，每个文本生成5个样本
# 用法: ./batch_gen_file.sh <gen_file> [output_dir] [num_samples]

# 配置参数
MODEL="F5TTS_v1_Base"
CKPT_FILE="ckpts/sft_data/model_14500.pt"
VOCAB_FILE="data/sft_data_pinyin/vocab.txt"
REF_AUDIO="data/sft_data/wavs/208.wav"
REF_TEXT="但方才那位老庄丁却说，床底下搜到的蛇是条<strong>白花</strong>蛇。"

# 输入文件（每行一个文本）
GEN_FILE="${1:-batch_gen_file_example.txt}"
# 输出目录
OUTPUT_DIR="${2:-outputs/batch_14500}"
# 每个文本生成的样本数
NUM_SAMPLES="${3:-5}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 计数器
counter=0

echo "开始批量生成..."
echo "输入文件: $GEN_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "每个文本生成样本数: $NUM_SAMPLES"
echo ""

# 读取文件，每行一个文本
while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行
    if [ -z "$line" ]; then
        continue
    fi
    
    # 去除可能的 voice 标签（如果存在）
    line=$(echo "$line" | sed 's/^\[[^]]*\]//')
    
    # 跳过处理后的空行
    if [ -z "$line" ]; then
        continue
    fi
    
    echo "=========================================="
    echo "处理第 $counter 个文本:"
    echo "文本: ${line:0:50}..."
    echo "将生成 $NUM_SAMPLES 个样本"
    echo "=========================================="
    
    # 为每个文本生成 NUM_SAMPLES 个样本
    for sample_idx in $(seq 0 $((NUM_SAMPLES - 1))); do
        # 生成输出文件名：batch_00000_sample0.wav, batch_00000_sample1.wav, ...
        output_file="batch_$(printf "%05d" $counter)_sample${sample_idx}.wav"
        output_path="$OUTPUT_DIR/$output_file"
        
        # 生成随机种子（基于 counter 和 sample_idx，确保可重复）
        seed=$((counter * 10000 + sample_idx * 1000 + 42))
        
        echo "  生成样本 $((sample_idx + 1))/$NUM_SAMPLES (seed: $seed)..."
        
        # 创建临时 Python 脚本来设置随机种子并调用 infer_cli
        python -c "
import sys
import torch
import numpy as np

# 设置随机种子
seed = $seed
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 设置命令行参数并调用 infer_cli
sys.argv = [
    'infer_cli.py',
    '--model', '$MODEL',
    '--ckpt_file', '$CKPT_FILE',
    '--vocab_file', '$VOCAB_FILE',
    '--ref_audio', '$REF_AUDIO',
    '--ref_text', '$REF_TEXT',
    '--gen_text', '$line',
    '--output_dir', '$OUTPUT_DIR',
    '--output_file', '$output_file',
    '--remove_silence'
]

# 导入并运行
sys.path.insert(0, 'src')
from f5_tts.infer.infer_cli import main
main()
" > /dev/null 2>&1
        
        # 检查是否成功
        if [ -f "$output_path" ]; then
            echo "  ✅ 已保存: $output_file"
            
            # 生成同名的 .lab 文件
            lab_file="${output_file%.wav}.lab"
            lab_path="$OUTPUT_DIR/$lab_file"
            
            # 清理文本：去掉 <strong> 和 </strong> 标签，去掉所有标点符号（包括中文标点）
            cleaned_text=$(echo "$line" | \
                sed 's/<strong>//g' | \
                sed 's/<\/strong>//g' | \
                sed 's/[，。！？；：、""''（）【】《》〈〉「」『』〖〗〘〙〚〛…—～·•]//g' | \
                sed 's/[[:punct:]]//g' | \
                sed 's/[[:space:]]\+/ /g' | \
                sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            
            # 写入 .lab 文件
            echo "$cleaned_text" > "$lab_path"
            echo "  ✅ 已保存: $lab_file"
        else
            echo "  ❌ 生成失败: $output_file"
        fi
    done
    
    echo ""
    
    # 增加计数器
    counter=$((counter + 1))
    
done < "$GEN_FILE"

echo "=========================================="
echo "批量生成完成！"
echo "共处理 $counter 个文本"
echo "每个文本生成 $NUM_SAMPLES 个样本"
echo "总样本数: $((counter * NUM_SAMPLES))"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
