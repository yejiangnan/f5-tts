import json
import re
import os
import shutil
import random
from pprint import pprint


def is_pure_chinese(text):
    """检查文本是否为纯中文（允许标点符号和空格）"""
    # 移除 <strong> 和 </strong> 标签
    text_without_tags = re.sub(r'<strong>|</strong>', '', text)
    # 检查是否包含英文、数字或其他非中文字符（除了中文标点、空格）
    # 中文字符范围：\u4e00-\u9fff
    # 中文标点：\u3000-\u303f, \uff00-\uffef
    pattern = r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s，。！？；：、""''（）【】《》]'
    return not bool(re.search(pattern, text_without_tags))


def extract_strong_words(text):
    """提取 <strong> </strong> 中包含的词"""
    pattern = r'<strong>(.*?)</strong>'
    matches = re.findall(pattern, text)
    return matches


def clean_text(text):
    """清理文本：移除 <strong> </strong> 外的混合内容，只保留纯中文"""
    # 先提取所有 <strong>...</strong> 标签及其内容
    strong_pattern = r'<strong>.*?</strong>'
    strong_matches = re.findall(strong_pattern, text)
    
    # 移除所有 <strong>...</strong> 标签，检查剩余文本
    text_without_strong = re.sub(strong_pattern, '', text)
    
    # 检查剩余文本是否为纯中文
    if is_pure_chinese(text_without_strong):
        return text  # 如果剩余部分是纯中文，保留原文本
    else:
        # 如果剩余部分不是纯中文，只保留 <strong>...</strong> 部分
        return ' '.join(strong_matches)



def clean_data(metadata_file, dpo_metadata_file, output_file):
    # 读取文件
    print(f"读取 {metadata_file}...")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"读取 {dpo_metadata_file}...")
    with open(dpo_metadata_file, 'r', encoding='utf-8') as f:
        dpo_metadata = json.load(f)
    
    # 处理每个条目
    cleaned_metadata = []
    for _, item_data in metadata.items():
        if len(item_data['info']) > 0:
            item_data['max_value'] = item_data['info'][-1][1]
            item_data["max_sample_id"] = item_data['info'][-1][0]   
            item_data["min_value"] = item_data['info'][0][1]
            item_data["min_sample_id"] = item_data['info'][0][0]

    for item in dpo_metadata:
        best_utt_id, best_sample_id = item["best_wav_path"].split("/")[-1].split(".")[0].split("_")
        best_sample_id = f"{best_utt_id}_{best_sample_id}"
        worst_utt_id, worst_sample_id = item["worst_wav_path"].split("/")[-1].split(".")[0].split("_")
        worst_sample_id = f"{worst_utt_id}_{worst_sample_id}"
        if best_utt_id in metadata and best_sample_id == metadata[best_utt_id]["max_sample_id"]:
            item["best_wav_score"] = metadata[best_utt_id]["max_value"]
        if worst_utt_id in metadata and worst_sample_id == metadata[worst_utt_id]["min_sample_id"]:
            item["worst_wav_score"] = metadata[worst_utt_id]["min_value"]

        if item["best_wav_score"] is None or item["best_wav_score"] < 1.5:
            continue

        # 检查文本：如果除了 <strong> </strong> 外有英文或阿拉伯数字，则跳过
        text = item.get("text", "")
        if not text:
            continue
        
        # 移除所有 <strong>...</strong> 标签
        text_without_strong = re.sub(r'<strong>.*?</strong>', '', text)
        
        # 检查是否包含英文或阿拉伯数字
        # 英文：a-zA-Z
        # 阿拉伯数字：0-9
        has_english_or_digit = bool(re.search(r'[a-zA-Z0-9]', text_without_strong))
        
        if not has_english_or_digit:
            cleaned_metadata.append(item)
    
    # 保存清理后的 metadata
    print(f"\n保存到 {output_file}...")
    print(f"cleaned_metadata length: {len(cleaned_metadata)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_metadata, f, ensure_ascii=False, indent=4)
    


def sample_wavs(metadata_file, wav_path, sample_output_path, k=50):
    if os.path.exists(sample_output_path):
        shutil.rmtree(sample_output_path)
    os.makedirs(sample_output_path)
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    sample_data = random.sample(metadata, k)
    
    # 根据 uttid 排序（从 best_wav_path 提取，如 xxx/00000_sample04.wav -> 00000）
    def get_uttid(item):
        basename = item["best_wav_path"].split("/")[-1].split(".")[0]  # e.g. 00000_sample04
        return basename.split("_")[0]  # uttid e.g. 00000
    sample_data.sort(key=get_uttid)
    
    for idx, item in enumerate(sample_data):
        wav_path = item["best_wav_path"]
        tgt_path = os.path.join(sample_output_path, wav_path.split("/")[-1])
        shutil.copy(wav_path, tgt_path)
        del sample_data[idx]["worst_wav_score"]
        del sample_data[idx]["worst_wav_path"]
        del sample_data[idx]["worst_duration_sec"]

    with open(os.path.join(sample_output_path, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # proc_dir = "stepf15_data/batch_1"
    proc_dir = "stepf06_data/batch_1"
    metadata_file = f"/data/F5-TTS/{proc_dir}/metadata.json"
    dpo_metadata_file = f"/data/F5-TTS/{proc_dir}/dpo_metadata.json"
    output_file = f"/data/F5-TTS/{proc_dir}/cleaned_metadata.json"

    wav_path = f"/data/F5-TTS/{proc_dir}/emphasis_wavs"
    sample_output_path = f"/data/F5-TTS/{proc_dir}/sample_wavs"

    clean_data(metadata_file, dpo_metadata_file, output_file)
    sample_wavs(output_file,wav_path, sample_output_path, k=50)
