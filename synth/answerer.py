# synth/answerer.py
from local_model.local_model_interface import generate
import json

# ==========================================
# 1. Initial Answer Prompt (Answerer)
# ==========================================
ANSWER_SYSTEM_PROMPT = """You are an expert-level Teaching Assistant proficient in University Physics. Your task is to provide logically rigorous and computationally precise solutions to physics problems.

Please strictly follow this standardized problem-solving process:
1. **Model Analysis**: Briefly describe the physical model involved and the force/energy analysis.
2. **Symbol Definition**: List the known quantities (values + units) and the target variables provided in the problem.
3. **Formula Derivation**: Establish equations based on physical theorems. Prioritize symbolic manipulation to derive the analytical expression for the target variable *before* substituting any numbers.
4. **Numerical Calculation**: Substitute values into the analytical expression. Pay attention to significant figures and unit conversion (use SI units uniformly).
5. **Final Result**: Clearly state the final answer and display it on a separate line in bold.

Problem:
"""

def answer_question(question: str, model_spec: str, max_tokens=1500, temp=0.3):
    # Slightly increased temp (0.3) for richer Chain of Thought; increased max_tokens for detailed steps
    prompt = f"{ANSWER_SYSTEM_PROMPT}{question}\n\nPlease begin your complete solution:"
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=temp)


# ==========================================
# 2. Self-Refinement Prompt (Refiner)
# ==========================================
REFINE_SYSTEM_PROMPT = """You are correcting a physics solution via a "Self-Reflection" mechanism.
You will be presented with: The Original Problem, Your Previous Solution, and a Review Report (in JSON format) from a strict Critic.

Please execute the following operations:
1. **Analyze the Review Report**:
   - If `critical_errors` is NOT empty: This means your previous solution contains **factual errors** (e.g., wrong formula, calculation error). You must **completely abandon** the erroneous path, re-think, and solve from scratch.
   - If there are only `suggestions`: This means the solution is generally correct but requires optimization in steps or logic.

2. **Execute Correction**:
   - Do not explain what you did wrong, and do not just write the corrected part.
   - **You must output a COMPLETE, INDEPENDENT, and CORRECTED solution**.
   - Maintain the same high-standard structure as the initial solution (Model Analysis -> Formula Derivation -> Numerical Calculation).

"""

def self_refine(question: str, current_answer: str, critic_feedback: str, model_spec: str, max_tokens=1500):
    # Build Prompt with clear separation of context
    prompt = (
        f"{REFINE_SYSTEM_PROMPT}\n"
        f"### Original Problem\n{question}\n\n"
        f"### Previous Solution\n{current_answer}\n\n"
        f"### Critic's Review Report (JSON)\n{critic_feedback}\n\n"
        f"### Corrected Complete Solution\n"  # Guide the model to start outputting the solution directly
    )
    
    # Temperature set to 0.1 to ensure the correction process is strictly rigorous and non-divergent
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=0.1)