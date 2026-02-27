#!/bin/bash

# Initialize conda for bash shell
eval "$(/data/miniforge3/bin/conda shell.bash hook)"

rm -rf /data/seed-tts-eval/outputs/grpo

# Activate conda environment
conda activate f5-tts

python src/f5_tts/infer/infer_seed.py \
    --model F5TTS_v1_Base \
    --ckpt_file /data/F5-TTS/ckpts/grpo_kl0.2_range0.1_noise0.1/model_1000.pt \
    --vocab_file data/sft_data_pinyin/vocab.txt \
    --vocoder_name vocos \
    --gen_file /data/seed-tts-eval/seedtts_testset/zh/meta.lst \
    --base_output_dir /data/seed-tts-eval/outputs/grpo \
    # --gen_duration 4.0

# Switch to seedeval environment for evaluation
conda activate seedeval
cd /data/seed-tts-eval
bash cal_wer.sh seedtts_testset/zh/meta.lst outputs/grpo/ zh
