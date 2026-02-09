#!/bin/bash



python src/f5_tts/infer/infer_batch.py \
    --model F5TTS_v1_Base \
    --ckpt_file /data/F5-TTS/ckpts/sft_data_dpo_2/model_15000.pt \
    --vocab_file data/sft_data_pinyin/vocab.txt \
    --ref_audio "/data/F5-TTS/prompts/stepf06.wav" \
    --ref_text "真的，我常想，要是当年有人能在我大一的时候<strong>亲自</strong>告诉我这些事就好了。" \
    --gen_file /data/F5-TTS/text_file/emphasis_text_format_batch6_f06.txt \
    --base_output_dir /data/F5-TTS/stepf06_data/batch_1/wavs \
    --count 20

# python src/f5_tts/infer/infer_batch.py \
#     --model F5TTS_v1_Base \
#     --ckpt_file /data/F5-TTS/ckpts/sft_data_dpo_2/model_15000.pt \
#     --vocab_file data/sft_data_pinyin/vocab.txt \
#     --ref_audio "/data/F5-TTS/prompts/stepf15.wav" \
#     --ref_text "所以，这种积极参与的方式，对我<strong>很多</strong>兴趣爱好的培养都起到了很大的作用。" \
#     --gen_file /data/F5-TTS/text_file/emphasis_text_format_batch7_f15.txt \
#     --base_output_dir /data/F5-TTS/stepf15_data/batch_1/wavs \
#     --count 20