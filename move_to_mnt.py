import os
import json
import shutil
from tqdm import tqdm


def move_to_mnt(proc_dir, dest_dir):
    base_dir = f"{dest_dir}/{proc_dir}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/wavs", exist_ok=True)

    metadata_file = f"{proc_dir}/cleaned_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    new_meta_data = []
    for item in tqdm(metadata):
        src_wav_path = item['best_wav_path'].replace("wavs", "emphasis_wavs")
        uttid = src_wav_path.split("/")[-1].split(".")[0]
        tgt_wav_path = f"{base_dir}/wavs/{uttid}.wav"
        shutil.copy(src_wav_path, tgt_wav_path)
        new_meta_data.append({
            "uttid": uttid,
            "wav_path": tgt_wav_path,
            "text": item['text'],
            "duration": item['best_duration_sec'],
        })
    with open(f"{base_dir}/metadata.json", 'w') as f:
        json.dump(new_meta_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    proc_dir = "stepf06_data/batch_1"
    dest_dir = f"/mnt/gpfs/yjn/emphasis_data/tts_sft_data"
    move_to_mnt(proc_dir, dest_dir)