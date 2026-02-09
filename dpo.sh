#!/bin/bash

# accelerate launch \
#     --num_processes=1 \
#     --mixed_precision=fp16 \
#     src/f5_tts/train/finetune_dpo.py \
#     --save_name emphasis_ids_enhanced_transformer \
#     --exp_name F5TTS_v1_Base \
#     --dataset_name sft_data \
#     --finetune \
#     --pretrain ckpts/F5TTS_v1_Base/model_1250000.safetensors \
#     --tokenizer pinyin \
#     --tokenizer_path data/sft_data_pinyin/vocab.txt \
#     --learning_rate 1e-5 \
#     --batch_size_per_gpu 3200 \
#     --batch_size_type frame \
#     --max_samples 64 \
#     --epochs 1000 \
#     --num_warmup_updates 500 \
#     --save_per_updates 500 \
#     --keep_last_n_checkpoints 100 \
#     --last_per_updates 150 \
#     --log_samples \
#     --logger tensorboard

accelerate launch \
    --num_processes=6 \
    --mixed_precision=fp16 \
    src/f5_tts/train/finetune_dpo.py \
    --save_name dpo_2 \
    --exp_name F5TTS_v1_Base \
    --dataset_name sft_data \
    --finetune \
    --pretrain /data/F5-TTS/ckpts/F5TTS_v1_Base/model_21500.safetensors\
    --tokenizer pinyin \
    --tokenizer_path data/sft_data_pinyin/vocab.txt \
    --learning_rate 1e-5 \
    --dpo_beta 500 \
    --sft_loss_weight 0.5 \
    --dpo_loss_weight 1 \
    --batch_size_per_gpu 3200 \
    --batch_size_type frame \
    --max_samples 64 \
    --epochs 500 \
    --num_warmup_updates 500 \
    --save_per_updates 500 \
    --keep_last_n_checkpoints 100 \
    --last_per_updates 150 \
    --log_samples \
    --logger tensorboard