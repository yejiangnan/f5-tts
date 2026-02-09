#!/bin/bash


python src/f5_tts/infer/infer_seed.py \
    --model F5TTS_v1_Base \
    --ckpt_file /data/F5-TTS/ckpts/F5TTS_v1_Base/model_1250000.safetensors \
    --vocab_file data/sft_data_pinyin/vocab.txt \
    --gen_file /data/seed-tts-eval/seedtts_testset/zh/meta.lst \
    --base_output_dir /data/seed-tts-eval/outputs/four_sec_1 \
    --gen_duration 4.0 \