# synth/critic.py
import os, json
from openai import OpenAI
from local_model.local_model_interface import generate

# 配置 OpenAI Client (Oracle)
base_url = "https://api.pumpkinaigc.online/v1"
client = OpenAI(
    api_key="sk-pew2UshHFpIXk3afA2B1C3A9F4A54250AfAc109c23D728B5",
    base_url=base_url
)

CRITIC_SYSTEM_PROMPT = """你是一个顶级的物理问题批评家。你的任务是严格验证一个解答是否正确。

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

def _call_openai(prompt, model):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Critic Error] OpenAI API call failed: {e}")
        return json.dumps({"critical_errors": [f"API Error: {e}"], "suggestions": [], "confidence": 0.0})

def _call_local(prompt, model_spec):
    # 本地模型调用，temperature=0.0 保证确定性
    return generate(model_spec, prompt, max_tokens=1024, temperature=0.0)

def verify_answer(answer: str, question: str, model_spec="gpt-4.1"):
    """
    执行一次 Critic 验证。
    返回包含 passed 状态、建议、以及用于 DPO 的原始文本和 Prompt。
    """
    # 1. 构建 Prompt
    prompt = CRITIC_SYSTEM_PROMPT.format(question=question, answer=answer)
    
    # 2. 调用模型
    # 简单的判断逻辑：如果是 openai model 或 gpt 开头，走 API；否则走本地
    if model_spec.startswith("gpt") or model_spec == "openai" or "gpt-4" in model_spec:
        raw_txt = _call_openai(prompt, model_spec)
    else:
        raw_txt = _call_local(prompt, model_spec)
        
    # 3. 解析 JSON
    clean_txt = raw_txt.strip()
    if clean_txt.startswith("```"):
        import re
        clean_txt = re.sub(r"^```(json)?", "", clean_txt).strip()
        clean_txt = re.sub(r"```$", "", clean_txt).strip()

    try:
        sc = json.loads(clean_txt)
    except Exception:
        sc = {"critical_errors": [f"JSON Parse Error"], "suggestions": [], "confidence": 0.0}
    
    critical_errors = sc.get("critical_errors", [])
    suggestions = sc.get("suggestions", [])
    passed = len(critical_errors) == 0
    all_issues = critical_errors + suggestions

    return {
        "passed": passed, 
        "issues": all_issues, 
        "critic_details": sc,
        "raw_output": raw_txt, # [关键] 用于 DPO 的 Chosen/Rejected 内容
        "prompt": prompt       # [关键] 用于 DPO 的 Input 内容
    }