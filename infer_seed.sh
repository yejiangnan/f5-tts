#!/bin/bash

# Initialize conda for bash shell
eval "$(/data/miniforge3/bin/conda shell.bash hook)"

rm -rf /data/seed-tts-eval/outputs/four_sec

# Activate conda environment
conda activate f5-tts

python src/f5_tts/infer/infer_seed.py \
    --model E2TTS_Base \
    --model_cfg /data/F5-TTS/src/f5_tts/configs/E2TTS_Base.yaml \
    --ckpt_file /home/i-yejiangnan/.cache/modelscope/hub/models/SWivid/E2-TTS_Emilia-ZH-EN/E2TTS_Base/model_1200000.pt \
    --vocab_file data/sft_data_pinyin/vocab.txt \
    --vocoder_name vocos \
    --gen_file /data/seed-tts-eval/seedtts_testset/zh/meta.lst \
    --base_output_dir /data/seed-tts-eval/outputs/four_sec \
    # --gen_duration 4.0

# Switch to seedeval environment for evaluation
conda activate seedeval
cd /data/seed-tts-eval
bash cal_wer.sh seedtts_testset/zh/meta.lst outputs/four_sec/ zh
