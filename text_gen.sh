#/bin/bash

python text_gen.py \
    --input_file "/data/F5-TTS/text_file/src_text_batch9.txt" \
    --save_file_path "/data/F5-TTS/grpo_data/emphasis_text_batch9.json" \
    --error_file_path "/data/F5-TTS/grpo_data/error.txt" \
    --max_workers 25 \
    --batch_size 50 \
    --rank 1 \
    --use_gpt4o "false" \
    --prompt_type "emphasis"