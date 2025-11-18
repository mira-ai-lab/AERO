# synth/answerer.py
from local_model.local_model_interface import generate

SYSTEM_PREFIX = "你是一个求解大学物理问题的助理，给出 step-by-step 的推导与最终数值（若有）。\n\n题目：\n"
def answer_question(question: str, model_spec: str, max_tokens=1024, temp=0.2):
    prompt = SYSTEM_PREFIX + question + "\n\n请给出完整解答（步骤清晰，若涉及计算给出数值与单位）。  "
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=temp)

def self_refine(question: str, current_answer: str, critic_feedback: str, model_spec: str, max_tokens=1024):
    prompt = (SYSTEM_PREFIX + question + "\n\n当前解答：\n" + current_answer +
              "\n\n批评家反馈：\n" + critic_feedback +
              "\n\n请基于上述反馈修正并给出最终、可验证的解答（保留步骤）。  ")
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=0.1)
