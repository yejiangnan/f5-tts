
from pathlib import Path


def main():
    src = Path("/mnt/gpfs/yjn/text_data_wash/zh_100w_ds_5.txt")
    dst = Path("/data/F5-TTS/text_file/src_text_batch5.txt")

    data = []
    with open(src, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(line)
    
    data = data[:50000]
    with open(dst, "w") as f:
        for line in data:
            f.write(line + "\n")


if __name__ == "__main__":
    main()