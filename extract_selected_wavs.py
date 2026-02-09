#!/usr/bin/env python3
"""从 cleaned_metadata.json 中提取匹配指定文本的 best_wav_path 音频到文件夹"""

import json
import shutil
from pathlib import Path

# 目标文本列表（文件中是 <strong> </strong>）
TARGET_TEXTS = [
    "没错，他们还有现炒的冰淇淋卷，压成块后淋上各种酱汁，<strong>确实</strong>有这种制作方式。",
    "但改为三角形结构后，受力会<strong>更加</strong>均匀。",
    "嗯，而且说<strong>真的</strong>，现在的人生规划根本规划不出个所以然，因为变数实在太多了。",
    "以上是他们的看法，不过我自己真正害怕不婚的原因<strong>到底</strong>是什么呢？",
    "二战后世界与战前截然不同。<strong>为何</strong>东南亚左翼主导的民族解放运动能如此迅速且大规模地兴起？",
    "他可能<strong>无法</strong>接受他人的批评，才会刻意经营人设。",
    "所以我就问他，<strong>为什么</strong>过了这么久才来？",
    "<strong>的确</strong>，很多北京人家习惯用绿豆煮汤或做点心，尤其像我们家这样。",
    "他啊，只要在电视、抖音或视频号上看到卖带鱼罐头的带货广告，<strong>必定</strong>见一次买一次。",
    "做料理这事儿啊，只要多花几道工序，味道<strong>绝对</strong>就提升一大截。",
]

def main():
    base_dir = Path(__file__).parent
    metadata_path = base_dir / "stepf15_data" / "batch_1" / "cleaned_metadata.json"
    output_dir = base_dir / "stepf15_data" / "batch_1" / "selected_text_wavs"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_set = set(TARGET_TEXTS)
    copied = 0
    found_texts = set()

    for item in data:
        text = item.get("text", "")
        if text in target_set:
            found_texts.add(text)
            wav_path = base_dir / item["best_wav_path"]
            if wav_path.exists():
                dest = output_dir / wav_path.name
                shutil.copy2(wav_path, dest)
                copied += 1
                print(f"已复制: {wav_path.name}")
            else:
                print(f"文件不存在: {wav_path}")

    not_found = set(TARGET_TEXTS) - found_texts
    print(f"\n完成！共复制 {copied} 个音频到 {output_dir}")
    if not_found:
        print(f"\n未匹配的文本 ({len(not_found)} 条):")
        for t in not_found:
            print(f"  - {t[:60]}...")

if __name__ == "__main__":
    main()
