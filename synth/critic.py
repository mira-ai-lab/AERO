# synth/critic.py
import os, json
from openai import OpenAI
from local_model.local_model_interface import generate
import re

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# 配置 OpenAI Client
base_url = "http://yy.dbh.baidu-int.com/v1"
client = OpenAI(
    api_key="sk-HsDcdnIrzLa2ywPYZsYESgsJhohPiw8SgvZ7zY8phJlARIeT",
    base_url=base_url
)

# 1. 主 Critic Prompt (S0 Check)
CRITIC_SYSTEM_PROMPT = """# Role
You are a Physics Competition Judge.

# Task
Verify the provided [Problem] and [Solution].

# Output
JSON format only.
- **critical_errors** (List[str]): Hard errors (logic/calc). Empty if correct.
- **suggestions** (List[str]): Optimization suggestions.
- **confidence** (float): 0.0-1.0.
- **correct_solution** (str or null): 
    - If errors found: Provide the COMPLETE correct derivation and FINAL ANSWER here. This is the Ground Truth.
    - If correct: null.

# Output Example(Please strictly adhere to the JSON output format.)
{{
  "critical_errors": ["Wrong integral limits..."],
  "suggestions": [],
  "confidence": 1.0,
  "correct_solution": "The correct derivation is...Final Answer: 42J"
}}

Problem:
{question}

Solution:
{answer}
"""

# 2. 等价性检查 Prompt (Cheap Check)
EQUIVALENCE_PROMPT = """# Role
You are a Physics TA.

# Task
Check if [Candidate] is equivalent to [Reference].
Focus on the FINAL RESULT and key logic. Allow minor format differences.

# Input
Problem: {question}
Reference (Gold): {ground_truth}
Candidate: {candidate}

# Output
JSON: {{"passed": true/false, "reason": "..."}}
"""

# 3. Gold-Guided Critique Prompt (Teacher Mode)
CRITIC_WITH_GOLD_PROMPT = """# Role
You are a Physics Tutor.

# Task
The Student's solution matches the Problem but differs from the Correct Answer.
Use the Correct Answer to write a Critique.
Point out the specific error in the Student's step WITHOUT giving away the full answer immediately. Guide them to self-correct.

# Input
Problem: {question}
Correct Answer: {ground_truth}
Student Solution: {answer}

# Output
JSON: {{"issues": ["..."], "suggestions": ["..."]}}
"""

def _call_openai(prompt, model, max_tokens=10240):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=40960
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Critic Error] API failed: {e}")
        return json.dumps({"critical_errors": [f"API Error"], "correct_solution": None})

def _call_local(prompt, model_spec, max_tokens=4096):
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=0.0)


def _parse_json(text):
    """
    增强版解析器：
    1. 剥离 Markdown 代码块
    2. 提取第一个 {...} JSON 对象
    3. 修复换行符和 LaTeX 转义
    """
    if not text:
        return None

    # --- 阶段 1: 提取 JSON 文本块 ---
    # 很多时候 LLM 会废话，我们需要找到最外层的 {}
    # 这是一个非贪婪匹配，寻找第一个 { 开始，直到最后一个 } 结束
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 如果找不到 {}，可能 LLM 没输出 JSON，或者格式全乱了
        # 尝试直接处理原文本（虽然希望能找到）
        json_str = text

    # --- 阶段 2: 字符串内容清洗 (Robust Logic) ---
    str_pattern = r'"((?:[^"\\]|\\.)*)"'
    
    def sanitize_string_content(match):
        content = match.group(1)
        # 1. LaTeX 反斜杠修复 ( \ -> \\ )
        content = content.replace('\\', '\\\\')
        # 2. 还原被误伤的转义引号 ( \\" -> \" )
        content = content.replace('\\\\"', '\\"') 
        # 3. 修复非法换行 ( \n -> \\n )
        content = content.replace('\n', '\\n')
        return f'"{content}"'

    sanitized_str = re.sub(str_pattern, sanitize_string_content, json_str, flags=re.DOTALL)

    # --- 阶段 3: 解析 ---
    try:
        return json.loads(sanitized_str)
    except json.JSONDecodeError:
        # 如果依然失败，尝试最后一种常见的容错：把 Markdown 的 ```json 去掉
        # 虽然阶段 1 应该处理了，但为了保险
        clean_text = re.sub(r'```json\s*', '', sanitized_str)
        clean_text = re.sub(r'```', '', clean_text)
        try:
            return json.loads(clean_text)
        except Exception as e:
            print(f"[Parse Error] Failed to parse: {str(e)}")
            # print(f"Raw Text snippet: {text[:100]}...") 
            return None

def verify_answer(answer: str, question: str, model_spec="deepseek-r1-250528"):
    """S0 深度验证，返回 Gold"""
    prompt = CRITIC_SYSTEM_PROMPT.format(question=question, answer=answer)
    
    is_api = any(x in model_spec for x in ["gpt-4", "openai", "deepseek"])
    raw = _call_openai(prompt, model_spec) if is_api else _call_local(prompt, model_spec)
    
    sc = _parse_json(raw)
    # 确保提取 correct_solution
    return {
        "passed": len(sc.get("critical_errors", [])) == 0,
        "issues": sc.get("critical_errors", []) + sc.get("suggestions", []),
        "critic_details": sc,
        "raw_output": raw,
        "prompt": prompt,
        "correct_solution": sc.get("correct_solution") # 关键字段
    }

def check_answer_equivalence(question: str, ground_truth: str, candidate: str, model_spec: str):
    """Local 模型进行廉价对比"""
    if not ground_truth: return {"passed": False, "reason": "No Gold"}
    
    prompt = EQUIVALENCE_PROMPT.format(question=question, ground_truth=ground_truth, candidate=candidate)
    raw = _call_local(prompt, model_spec, max_tokens=512)
    res = _parse_json(raw)
    return {"passed": res.get("passed", False), "reason": res.get("reason", "Parse Error")}

def generate_critique_with_gold(question: str, answer: str, ground_truth: str, model_spec: str):
    """Teacher Mode: 看着答案给 Feedback"""
    prompt = CRITIC_WITH_GOLD_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer)
    raw = _call_local(prompt, model_spec, max_tokens=1024)
    res = _parse_json(raw)
    issues = res.get("issues", []) + res.get("suggestions", [])
    return {"issues": issues, "raw_output": raw, "prompt": prompt}