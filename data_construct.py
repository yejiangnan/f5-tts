
import json
import shutil
from pathlib import Path
from pprint import pprint


def main():
    meta_data = []
    wavs_path = Path("/data/datasets/emphasis_data/zh/train/batch_0_wavs_16k")
    json_path = Path("/data/datasets/emphasis_data/zh/train/batch_0_selected_tagged.json")
    json_data = json.loads(json_path.read_text())

    wavs_dict = {}
    for wav in wavs_path.glob("*.wav"):
        uttid = wav.stem.split("_")[1]
        wavs_dict[uttid] = f"cosy1_batch0/{str(wav).split('/')[-1]}"

    for item in json_data:
        if item['tag'] == "是":
            meta_data.append({
                "audio_path": wavs_dict[item['uttid']],
                "text": item['text'],
            })

    wavs_path = Path("/data/datasets/emphasis_data/zh/train/batch_1_wavs_16k")
    json_path = Path("/data/datasets/emphasis_data/zh/train/batch_1_selected_tagged.json")
    json_data = json.loads(json_path.read_text())

    wavs_dict = {}
    for wav in wavs_path.glob("*.wav"):
        uttid = wav.stem.split("_")[1]
        wavs_dict[uttid] = f"cosy1_batch1/{str(wav).split('/')[-1]}"

    for item in json_data:
        if item['tag'] == "是":
            meta_data.append({
                "audio_path": wavs_dict[item['uttid']],
                "text": item['text'],
            })
    
    with open("data/sft_data/metadata.csv", "w") as f:
        for item in meta_data:
            f.write(f"{item['audio_path']}|{item['text']}\n")

def convert_fomat():
    with open("data/sft_data/metadata.csv", "r") as f:
        lines = f.readlines()
    
    metadata = []
    
    for idx, line in enumerate(lines):
        audio_path, text = line.strip().split("|")
        shutil.copy(f"/data/F5-TTS/data/sft_data/{audio_path}", f"data/sft_data/wavs/{idx}.wav")
        metadata.append((f"wavs/{idx}.wav", text))
    
    with open("data/sft_data/new_metadata.csv", "w") as f:
        for item in metadata:
            f.write(f"{item[0]}|{item[1]}\n")

if __name__ == "__main__":
    # main()
    convert_fomat()


    