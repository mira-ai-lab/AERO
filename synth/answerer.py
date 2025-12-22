# synth/answerer.py
from local_model.local_model_interface import generate
import json

# ==========================================
# 1. Answer Prompt (Updated)
# ==========================================
ANSWER_SYSTEM_PROMPT = """You are an expert-level Teaching Assistant with strong background in mathematics and quantitative sciences. 
Your task is to provide logically rigorous, well-structured, and computationally precise solutions to advanced academic problems.

Please reason step by step, and present your final answer clearly within \\boxed{} when an explicit result is required.

Please strictly follow this standardized problem-solving process:
1. **Problem Analysis**: Clarify the problem setting, assumptions, and the underlying model or framework.
2. **Symbol Definition**: Clearly define all given quantities, parameters, variables, and target unknowns.
3. **Derivation / Reasoning**: Develop the solution through logical arguments, proofs, or analytical derivations.
4. **Computation / Simplification**: Carry out necessary calculations or symbolic manipulations.
5. **Final Result**: State the final conclusion or expression explicitly, wrapping the main result in \\boxed{} when appropriate.
   Example: The final value is \\boxed{42}.

Problem:
"""


def answer_question(question: str, model_spec: str, max_tokens=4096, temp=1.0):
    prompt = f"{ANSWER_SYSTEM_PROMPT}{question}\n\nPlease begin your complete solution:"
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=temp)


# ==========================================
# 2. Self-Correction / Attack Prompt
# ==========================================
ATTACK_SYSTEM_PROMPT = """You are a rigorous Academic Reviewer with expertise in mathematics, physics, and quantitative sciences.  
You are provided with a **Problem** and a **Candidate Solution**, which is suspected to be **INCORRECT**.

Your Task:
1. Begin with the assumption that the Candidate Solution contains an error.
2. Carefully examine the logical reasoning, definitions, assumptions, derivations, and calculations.
3. Identify the precise flaw or unjustified step (the error may be subtle or conceptual).
4. **Re-solve the problem from first principles**, using a clear and logically sound approach.
5. Present the corrected result clearly, and **wrap the final answer in \\boxed{}** when an explicit result is required.

Output Format:
Thinking Process: <Analyze where the error or weakness occurs>
Correct Solution: <Complete and rigorous derivation or reasoning>
Final Answer: \\boxed{<Corrected result>}
"""

def self_correct(question: str, bad_answer: str, model_spec: str, max_tokens=4096):
    prompt = (
        f"{ATTACK_SYSTEM_PROMPT}\n"
        f"### Problem\n{question}\n\n"
        f"### Suspected Incorrect Solution\n{bad_answer}\n\n"
        f"### Your Correction\n"
    )
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=0.1)