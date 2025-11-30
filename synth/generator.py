# synth/generator.py
import os, json
from local_model.local_model_interface import generate
from utils.io import read_text

PROMPT_FILE = "synth/prompt_template.txt"

def load_prompt_template():
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"{PROMPT_FILE} not found.")
    return read_text(PROMPT_FILE)

def generate_questions(n: int, model_spec: str, temperature=0.9, max_tokens=1024):
    prompt_template = load_prompt_template()
    questions = []
    for i in range(n):
        # prompt_text = prompt_template.replace("{focus}", "综合") + f"\n# id:{i}\n"
        prompt_text = prompt_template
        out = generate(model_spec, prompt_text, max_tokens=max_tokens, temperature=temperature)
        try:
            data = json.loads(out.strip())
            questions.append({
                "question": data.get("question", out), 
                "meta": data.get("meta", {}),
                "prompt": prompt_text
            })
        except Exception:
            questions.append({
                "question": out.strip(), 
                "meta": {},
                "prompt": prompt_text
            })
    return questions