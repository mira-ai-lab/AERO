import os, re, json
import sympy as sp
from openai import OpenAI
base_url = "https://api.pumpkinaigc.online/v1"

client = OpenAI(
    api_key="sk-pew2UshHFpIXk3afA2B1C3A9F4A54250AfAc109c23D728B5",
    base_url=base_url
)

def self_check_llm(answer: str, question: str, openai_model="gpt-4.1"):
    """
    1. 强制 LLM 扮演 Oracle，先自己解题，再对比。
    2. 结构化输出 critical_errors (事实错误) 和 suggestions (建议)。
    """
    prompt = f"""你是一个顶级的物理问题批评家。你的任务是严格验证一个解答是否正确。

题目：
{question}

待评估的解答：
{answer}

请遵循以下步骤：
1.  **[内部思考]** 首先，请在内部（不要在最终输出中展示）完整地解答一遍这个问题，得到一个标准答案（包括关键步骤、公式和最终数值与单位）。
2.  **[对比验证]** 然后，将“待评估的解答”与你的“标准答案”进行严格对比。
3.  **[输出报告]** 最后，以 JSON 格式输出你的报告。

你的报告必须包含三个字段：
- "critical_errors": 列表。只包含**事实性错误**，例如：错误的公式、错误的数值计算、错误的单位、与题目条件不符的推导。如果此列表为空，代表解答在事实上是正确的。
- "suggestions": 列表。包含非事实性的改进建议，例如：逻辑跳跃、步骤不够清晰、格式问题。
- "confidence": 浮点数（0.0到1.0），表示你对“critical_errors”列表准确性的信心。

JSON 格式：
{{"critical_errors": ["..."], "suggestions": ["..."], "confidence": 0.0}}

请直接输出 JSON，不要包含其他任何文本。
"""
    
    try:
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        txt = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Critic Error] OpenAI API call failed: {e}")
        return {"critical_errors": [f"API Error: {e}"], "suggestions": [], "confidence": 0.0}

    try:
        return json.loads(txt)
    except Exception:
        return {"critical_errors": [f"JSON Parse Error: {txt}"], "suggestions": [], "confidence": 0.0}

# ========================================================================
# [说明] 以下两个函数 (extract_numeric_expressions, oracle_check_with_sympy)
# 在新的 verify_answer 逻辑中不再被调用，因为 LLM Critic 的职责已包含计算验证。
# 保留它们以供将来可能的其他用途。
# ========================================================================

def extract_numeric_expressions(answer_text: str):
    exprs = []
    # 匹配 $...$ 或 simple numbers
    for m in re.findall(r"\$(.+?)\$", answer_text, flags=re.S):
        exprs.append(m.strip())
    for m in re.findall(r"([-+]?\d+\.\d+|\d+)(?:\s*[a-zA-Z/%°μΩohm]*)", answer_text):
        exprs.append(m)
    return exprs

def oracle_check_with_sympy(expr: str):
    try:
        parsed = sp.sympify(expr)
        val = float(parsed.evalf())
        return {"ok": True, "value": val}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def verify_answer(answer: str, question: str, openai_model="gpt-4.1"):
    """
    1. 完全依赖 self_check_llm 的结构化输出来判断。
    2. 移除原有的 SymPy 检查，因为它已被新的 Prompt 逻辑取代。
    3. "passed" 仅当 "critical_errors" 列表为空时才为 True。
    """
    
    # 1. 调用改进后的 LLM Critic
    sc = self_check_llm(answer, question, openai_model=openai_model)
    
    critical_errors = sc.get("critical_errors", [])
    suggestions = sc.get("suggestions", [])
    
    # 2. [新逻辑] 仅当没有严重错误时，才算通过
    passed = len(critical_errors) == 0
    
    # 3. 组合所有反馈，用于 inner_loop 中的 self_refine
    #    (即使通过了，也可能包含 suggestions)
    all_issues = critical_errors + suggestions

    # 4. 返回 inner_loop.py 所需的格式
    #    "issues" 字段将用于 "self_refine" 的提示
    return {
        "passed": passed, 
        "issues": all_issues, 
        "critic_details": sc # 存储完整的结构化报告以供分析
    }