#!/usr/bin/env bash
set -euo pipefail

WAV_DIR=${1:-/data/F5-TTS/paper_eval/dpo/wavs}
ANNOTATION_DIR=${2:-/data/F5-TTS/paper_eval/dpo/aligned}
JSON_OUTPUT=${3:-/data/F5-TTS/paper_eval/dpo/prominence.json}
WAV_OUT_DIR=${4:-/data/F5-TTS/paper_eval/dpo/emphasis_wavs}
NUM_WORKERS=${5:-8}  # 可选的并行进程数

if [ -z "$NUM_WORKERS" ]; then
    python wavelet_prosody_toolkit/compute_prominence.py \
      --wav_dir "$WAV_DIR" \
      --annotation_dir "$ANNOTATION_DIR" \
      --json_output "$JSON_OUTPUT" \
      --wav_out_dir "$WAV_OUT_DIR"
else
    python wavelet_prosody_toolkit/compute_prominence.py \
      --wav_dir "$WAV_DIR" \
      --annotation_dir "$ANNOTATION_DIR" \
      --json_output "$JSON_OUTPUT" \
      --wav_out_dir "$WAV_OUT_DIR" \
      --num_workers "$NUM_WORKERS"
fi
