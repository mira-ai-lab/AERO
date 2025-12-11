# synth/answerer.py
from local_model.local_model_interface import generate
import json

# ==========================================
# 1. Answer Prompt (Updated)
# ==========================================
ANSWER_SYSTEM_PROMPT = """You are an expert-level Teaching Assistant proficient in University Physics. 
Your task is to provide logically rigorous and computationally precise solutions to physics problems.

Please reason step by step, and put your final answer within \\boxed{}.

Please strictly follow this standardized problem-solving process:
1. **Model Analysis**: Briefly describe the physical model involved.
2. **Symbol Definition**: List known quantities and target variables.
3. **Formula Derivation**: Establish equations and derive the analytical expression.
4. **Numerical Calculation**: Substitute values.
5. **Final Result**: Ensure the final answer is explicitly wrapped in \\boxed{}.
   Example: The final energy is \\boxed{42 J}.

Problem:
"""

def answer_question(question: str, model_spec: str, max_tokens=4096, temp=0.7):
    prompt = f"{ANSWER_SYSTEM_PROMPT}{question}\n\nPlease begin your complete solution:"
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=temp)


# ==========================================
# 2. Self-Correction / Attack Prompt
# ==========================================
ATTACK_SYSTEM_PROMPT = """You are a rigorous Physics Reviewer. 
You are provided with a Problem and a Candidate Solution.
The Candidate Solution is suspected to be **INCORRECT**.

Your Task:
1. Assume the Candidate Solution is wrong.
2. Carefully check the derivation steps, physical principles applied, and calculations.
3. Find the flaw (it might be subtle).
4. **Solve the problem again from scratch** to provide the correct solution.
5. You MUST put your new final answer within \\boxed{}.

Output Format:
Thinking Process: <Analyze where the error might be>
Correct Solution: <Full derivation>
Final Answer: \\boxed{<The corrected result>}
"""

def self_correct(question: str, bad_answer: str, model_spec: str, max_tokens=4096):
    prompt = (
        f"{ATTACK_SYSTEM_PROMPT}\n"
        f"### Problem\n{question}\n\n"
        f"### Suspected Incorrect Solution\n{bad_answer}\n\n"
        f"### Your Correction\n"
    )
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=0.1)