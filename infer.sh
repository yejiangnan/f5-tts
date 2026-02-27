#!/bin/bash


# f5-tts_infer-cli --model F5TTS_v1_Base \
#     --ref_audio "/data/datasets/emphasis_data/zh/steptts/batch_0/wavs/00001_step_f15.wav" \
#     --ref_text "生字词基础掌握得不错，接下来可以适当提升难度并增加一些训练量。" \
#     --gen_text "你好， 今天天气真不错！"
    # --gen_text "他<strong>到底</strong>想要干什么？" \

python src/f5_tts/infer/infer_cli.py \
    --model F5TTS_v1_Base \
    --ckpt_file /data/F5-TTS/ckpts/grpo_no_kl/model_2400.pt \
    --vocab_file data/sft_data_pinyin/vocab.txt \
    --ref_audio "/data/F5-TTS/data/sft_data/wavs/712.wav" \
    --ref_text "不会做付费项目。我们的初衷是让更多人能看到喜欢的剧集，这也是能坚持五年多的<strong>根本</strong>原因。虽然更新不频繁，但始终保持着分享的热情。" \
    --gen_text "这种行为是<strong>绝对</strong>不能被原谅的。" \
    --output_file infer_2400.wav