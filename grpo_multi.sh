#!/bin/bash

MAIN_NODE_IP="${MASTER_ADDR}"
MAIN_PORT=29500
NUM_NODES="${NODE_COUNT}"
GPUS_PER_NODE="${PROC_PER_NODE}"
TOTAL_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))
MACHINE_RANK="${NODE_RANK}"

export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

accelerate launch \
    --num_processes ${TOTAL_PROCESSES} \
    --num_machines ${NUM_NODES} \
    --machine_rank ${MACHINE_RANK} \
    --main_process_ip ${MAIN_NODE_IP} \
    --main_process_port ${MAIN_PORT} \
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
