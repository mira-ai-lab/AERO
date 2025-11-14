# synth/critic.py
import os, re, json
import sympy as sp
from openai import OpenAI
# openai.api_key = os.environ.get("OPENAI_API_KEY", "sk-pew2UshHFpIXk3afA2B1C3A9F4A54250AfAc109c23D728B5")
# 客户端将自动从环境变量 OPENAI_API_KEY 读取密钥
base_url = "https://api.pumpkinaigc.online/v1"

client = OpenAI(
    api_key="sk-pew2UshHFpIXk3afA2B1C3A9F4A54250AfAc109c23D728B5",
    base_url=base_url
)

def self_check_llm(answer: str, question: str, openai_model="gpt-4.1"):
    prompt = f"""你是物理题批评家（精通计算与量纲检验）。题目：{question}
解答：
{answer}

请指出该解答中的逻辑错误、计算错误、单位或量纲问题；列出发现的问题点（短句列表）。请以 JSON 输出：{{"issues": [...], "confidence": 0.0}}"""
    
    try:
        # [修改] 使用新版 client.chat.completions.create 语法
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        # [修改] 响应的结构也变了
        txt = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Critic Error] OpenAI API call failed: {e}")
        return {"issues": [f"API Error: {e}"], "confidence": 0.0}

    try:
        return json.loads(txt)
    except Exception:
        return {"issues": [txt], "confidence": 0.0}

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

def verify_answer(answer: str, question: str, openai_model="gpt-4"):
    issues = []
    sc = self_check_llm(answer, question, openai_model=openai_model)
    if sc.get("issues"):
        issues.extend(sc["issues"])
    exprs = extract_numeric_expressions(answer)
    oracle_results = []
    for e in exprs:
        r_sym = oracle_check_with_sympy(e)
        oracle_results.append({"expr": e, "sympy": r_sym})
    passed = len(issues) == 0 and all(r.get("sympy", {}).get("ok", True) for r in oracle_results)
    return {"passed": passed, "issues": issues, "oracle_results": oracle_results}
