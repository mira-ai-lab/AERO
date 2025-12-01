# synth/generator.py
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed # 新增
from local_model.local_model_interface import generate
from utils.io import read_text

PROMPT_FILE = "synth/prompt_template.txt"

def load_prompt_template():
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"{PROMPT_FILE} not found.")
    return read_text(PROMPT_FILE)

# 新增 max_workers 参数，默认为 10
def generate_questions(n: int, model_spec: str, temperature=0.9, max_tokens=1024, max_workers=10):
    prompt_template = load_prompt_template()
    questions = []
    
    def _generate_single(_):
        # prompt_text = prompt_template.replace("{focus}", "综合") 
        prompt_text = prompt_template
        
        try:
            out = generate(model_spec, prompt_text, max_tokens=max_tokens, temperature=temperature)
            data = json.loads(out.strip())
            return {
                "question": data.get("question", out), 
                "meta": data.get("meta", {}),
                "prompt": prompt_text
            }
        except Exception:
            # 容错处理
            return {
                "question": out.strip() if 'out' in locals() else "", 
                "meta": {},
                "prompt": prompt_text
            }

    print(f"Generating {n} questions with {max_workers} workers...")
    
    # 使用线程池并发请求
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交 n 个任务
        futures = [executor.submit(_generate_single, i) for i in range(n)]
        
        for future in as_completed(futures):
            try:
                res = future.result()
                if res["question"]: # 简单过滤空结果
                    questions.append(res)
            except Exception as e:
                print(f"Generation task failed: {e}")

    return questions