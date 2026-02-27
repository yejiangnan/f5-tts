#!/bin/bash
# 单机多卡训练（当前为 8 GPU）
# 多机训练请使用 grpo_multinode.sh，需配置 MAIN_NODE_IP、MACHINE_RANK 等

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

export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
accelerate launch \
    --num_processes=8 \
    --mixed_precision=fp16 \
    src/f5_tts/train/finetune_grpo.py \
    --save_name grpo \
    --exp_name F5TTS_v1_Base \
    --dataset_name sft_data \
    --finetune \
    --pretrain /data/F5-TTS/ckpts/grpo_kl08/model_1000.pt \
    --checkpoint_dir ckpts/grpo \
    --tokenizer pinyin \
    --tokenizer_path data/sft_data_pinyin/vocab.txt \
    --learning_rate 1e-4 \
    --batch_size_per_gpu 4 \
    --batch_size_type sample \
    --max_samples 64 \
    --epochs 1000 \
    --num_warmup_updates 100 \
    --save_per_updates 200 \
    --keep_last_n_checkpoints 100 \
    --last_per_updates 150 \
    --log_samples \
    --logger tensorboard \
    --beta 0.2 \
    --clip_range 0.1
# 推理时建议 load_model(..., use_ema=True)，使用 EMA 权重可减少抖动、提升听感