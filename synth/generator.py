import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from local_model.local_model_interface import generate
from utils.io import read_text
from tqdm import tqdm
import re

PROMPT_FILE = "synth/prompt_template.txt"

def load_prompt_template():
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"{PROMPT_FILE} not found.")
    return read_text(PROMPT_FILE)

def generate_questions(n: int, model_spec: str, temperature=1.0, max_tokens=1024, max_workers=30):
    raw_template = load_prompt_template()
    questions = []

    def _generate_single(idx):
        prompt_text = raw_template 

        try:
            out = generate(model_spec, prompt_text, max_tokens=max_tokens, temperature=temperature)
            data = json.loads(out.strip())
            return {
                "question": data.get("question", out), 
                "meta": data.get("meta", {}),
                "prompt": prompt_text
            }
        except Exception:
            return {
                "question": out.strip() if 'out' in locals() else "", 
                "meta": {},
                "prompt": prompt_text
            }
    
    print(f"Generating {n} questions with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_generate_single, i) for i in range(n)]
        
        for future in tqdm(as_completed(futures), total=n, desc="Generating Questions"):
            try:
                res = future.result()
                if res["question"]:
                    questions.append(res)
            except Exception as e:
                print(f"Generation task failed: {e}")

    return questions