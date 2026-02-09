#/bin/bash

python text_gen.py \
    --input_file "/data/F5-TTS/text_file/src_text_batch5.txt" \
    --save_file_path "/data/F5-TTS/text_file/emphasis_text_batch5.json" \
    --error_file_path "/data/F5-TTS/text_file/error.txt" \
    --max_workers 25 \
    --batch_size 50 \
    --rank 1 \
    --use_gpt4o "false" \
    --prompt_type "emphasis"