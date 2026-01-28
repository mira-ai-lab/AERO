# utils/make_kto_data.py
import json
import os
from utils.io import read_jsonl

def convert_to_kto_format(infile, outfile):
    data = read_jsonl(infile)
    out = []
    for item in data:
        # item: {"prompt": "...", "completion": "...", "label": True/False}
        out.append({
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]}
            ],
            "label": item["label"]
        })
        
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(out)} items to KTO format: {outfile}")

if __name__ == "__main__":
    if os.path.exists("kto_data/kto_data.jsonl"):
        convert_to_kto_format("kto_data/kto_data.jsonl", "kto_data/kto_final.json")