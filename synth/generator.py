# synth/generator.py
import os, json
from local_model.local_model_interface import generate
from utils.io import read_text, write_text

PROMPT_FILE = "synth/prompt_template.txt"

def load_prompt_template():
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"{PROMPT_FILE} not found.")
    return read_text(PROMPT_FILE)

def generate_questions(n: int, model_spec: str, temperature=0.9, max_tokens=1024):
    prompt_template = load_prompt_template()
    questions = []
    for i in range(n):
        prompt = prompt_template.replace("{focus}", "综合") + f"\n# id:{i}\n"
        out = generate(model_spec, prompt, max_tokens=max_tokens, temperature=temperature)
        try:
            data = json.loads(out.strip())
            questions.append({"question": data.get("question", out), "meta": data.get("meta", {})})
        except Exception:
            # 如果返回不能解析为 JSON，直接把文本作为 question 字段
            questions.append({"question": out.strip(), "meta": {}})
    return questions
