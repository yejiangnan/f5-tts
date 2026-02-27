import json
import re
from pprint import pprint


def main():
    prominence_json = "/data/F5-TTS/paper_eval/dpo/prominence.json"
    text_file = "/data/F5-TTS/test.txt"

    data = []
    with open(prominence_json, "r") as f:
        json_data = json.load(f)

    text_data = []
    with open(text_file, "r") as f:
        for line in f:
            text_data.append(line.strip())
    
    data = []
    for (key, val), text_item in zip(json_data.items(), text_data):
        data.append({
            "prominence": val,
            "text": text_item,
        })

    scores = []
    for item in data:
        text = item["text"]
        emphasis_words = re.findall(r"<strong>(.*?)</strong>", text)[0]
        # print(text)
        # print(item["prominence"]["words"])
        # print("---------------------------")
        # continue
        s = 0
        for word in item["prominence"]["words"]:
            w, score = word.split(":")
            score = float(score)
            if w in emphasis_words or emphasis_words in w:
                s = max(s, score)
        scores.append(s)

    print(scores)
    print(f"Average score: {sum(scores) / len(scores)}")



if __name__ == "__main__":
    main()