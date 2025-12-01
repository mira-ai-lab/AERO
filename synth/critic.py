# synth/critic.py
import os, json
from openai import OpenAI
from local_model.local_model_interface import generate
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
# 配置 OpenAI Client (Oracle)
# os.environ['https_proxy'] = 'http://agent.baidu.com:8891'
base_url = "http://yy.dbh.baidu-int.com/v1"
client = OpenAI(
    api_key="sk-HsDcdnIrzLa2ywPYZsYESgsJhohPiw8SgvZ7zY8phJlARIeT",
    base_url=base_url
)

CRITIC_SYSTEM_PROMPT = """# Role
You are a Physics Competition Judge and Verification Engine with the highest level of academic rigor. Your core competency lies in detecting logical flaws, calculation errors, and dimensional inconsistencies in physics derivations.

# Task
Please strictly verify the correctness of the provided [Problem] and [Solution under Evaluation].

# Workflow (Must be strictly followed)
1.  **Independent Derivation (Implicit Step)**: Before generating any output, you must independently and completely solve the problem in the background to establish an absolute "Ground Truth." Ensure your derivation covers force analysis, theorem application, calculus operations, and dimensional analysis.
2.  **Discrepancy Analysis**: Compare every step, formula citation, intermediate value, and final result of the [Solution under Evaluation] against your "Ground Truth" line by line.
3.  **Verdict Report**: Output the review report strictly in JSON format.

# Output Fields Definition
-   **critical_errors** (List[str]): Contains only **hard errors** that lead to an incorrect answer or a collapse of physical logic.
    -   Includes: incorrect application of principles, unmet conditions for formulas, mathematical calculation errors, unit/dimensional errors, or incorrect substitution of initial values.
    -   If the solution is factually correct, this list must be empty `[]`.
-   **suggestions** (List[str]): Contains **soft suggestions** for optimizing the solution process.
    -   Includes: excessive logical jumps, non-standard notation, redundant steps, or the existence of a more elegant solution.
-   **confidence** (float): A value between 0.0 and 1.0, representing your confidence level in the verdict (specifically regarding the presence or absence of "critical_errors").

# JSON Output Example
{{
  "critical_errors": [
    "...",
    "..."
  ],
  "suggestions": [
    "...",
    "..."
  ],
  "confidence": 0.98
}}

# Format Constraints
-   The output must be valid JSON format.
-   **Strictly NO** Markdown code block markers (e.g., ```json).
-   **Strictly NO** preambles, postscripts, or thinking processes outside the JSON object.

# Input Data
Problem:
{question}

Solution under Evaluation:
{answer}
"""

def _call_openai(prompt, model):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=10240
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Critic Error] OpenAI API call failed: {e}")
        return json.dumps({"critical_errors": [f"API Error: {e}"], "suggestions": [], "confidence": 0.0})

def _call_local(prompt, model_spec):
    # 本地模型调用，temperature=0.0 保证确定性
    return generate(model_spec, prompt, max_tokens=10240, temperature=0.0)

def verify_answer(answer: str, question: str, model_spec="deepseek-r1-250528"):
    """
    执行一次 Critic 验证。
    返回包含 passed 状态、建议、以及用于 DPO 的原始文本和 Prompt。
    """
    # 1. 构建 Prompt
    prompt = CRITIC_SYSTEM_PROMPT.format(question=question, answer=answer)
    
    # 2. 调用模型
    # 简单的判断逻辑：如果是 openai model 或 gpt 开头，走 API；否则走本地
    is_api_model = (
        model_spec.startswith("gpt") or 
        model_spec == "openai" or 
        "gpt-4" in model_spec or       # 注意这里加了 in model_spec
        "deepseek-r1" in model_spec
    )
    if is_api_model:
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