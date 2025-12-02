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
    if not text:
        return None

    # --- 1. 智能提取 (支持截断修复) ---
    def extract_candidate_block(s, start_idx):
        stack = []
        in_string = False
        escape = False
        
        # 记录最后一次有效字符的位置
        last_valid_char_index = -1
        
        for i, char in enumerate(s[start_idx:], start=start_idx):
            last_valid_char_index = i
            
            # 转义符处理
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            
            # 引号处理
            if char == '"':
                in_string = not in_string
                continue
            
            # 如果在字符串内，忽略结构字符
            if in_string:
                continue
            
            # 堆栈计数
            if char == '{':
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        # 完美闭合
                        return s[start_idx : i + 1]
        
        # === 新增逻辑：处理截断 (Truncation Handling) ===
        # 如果循环结束了，栈还不为空，说明 JSON 没写完
        if stack:
            print("[Debug] 检测到非闭合的 JSON，尝试自动修复...")
            partial_str = s[start_idx : last_valid_char_index + 1]
            
            # 修复 1: 如果还在字符串内，先闭合字符串
            if in_string:
                partial_str += '"'
            
            # 修复 2: 根据栈的深度，补全所有的 '}'
            partial_str += '}' * len(stack)
            
            return partial_str
            
        return None

    # --- 2. 深度清洗 (LaTeX & 换行符) ---
    def clean_json_string(raw_json):
        def clean_content(match):
            content = match.group(1)
            # 1. 保护性替换：把所有 \ 变成 \\
            content = content.replace('\\', '\\\\')
            # 2. 还原被误伤的转义引号 ( \\" -> \" )
            content = content.replace('\\\\"', '\\"')
            # 3. 修复真实换行符
            content = content.replace('\n', '\\n').replace('\r', '')
            # 4. 修复 Tab
            content = content.replace('\t', '\\t')
            return f'"{content}"'

        # 正则：匹配双引号内的内容
        str_pattern = re.compile(r'"((?:[^"\\]|\\.)*)"', re.DOTALL)
        return str_pattern.sub(clean_content, raw_json)

    # --- 主循环 ---
    cursor = 0
    while True:
        start_index = text.find('{', cursor)
        if start_index == -1:
            print("--- 未找到可解析的 JSON ---")
            return None
        
        # 提取（包含修复逻辑）
        candidate = extract_candidate_block(text, start_index)
        
        if candidate:
            # 清洗
            cleaned = clean_json_string(candidate)
            try:
                # 解析
                return json.loads(cleaned, strict=False)
            except json.JSONDecodeError as e:
                # 如果修复后还是解析不了（比如中间缺逗号），就放弃这个块，找下一个
                # print(f"[Debug] 解析尝试失败: {e}")
                pass
        
        cursor = start_index + 1

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
    raw = _call_local(prompt, model_spec, max_tokens=2048)
    res = _parse_json(raw)
    return {"passed": res.get("passed", False), "reason": res.get("reason", "Parse Error")}

def generate_critique_with_gold(question: str, answer: str, ground_truth: str, model_spec: str):
    """Teacher Mode: 看着答案给 Feedback"""
    prompt = CRITIC_WITH_GOLD_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer)
    raw = _call_local(prompt, model_spec, max_tokens=2048)
    res = _parse_json(raw)
    issues = res.get("issues", []) + res.get("suggestions", [])
    return {"issues": issues, "raw_output": raw, "prompt": prompt}