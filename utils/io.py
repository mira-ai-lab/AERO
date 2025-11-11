# utils/io.py
import json
from typing import List, Dict

def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def write_jsonl(path: str, records: List[Dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_text(path: str, s: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(s)