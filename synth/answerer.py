from local_model.local_model_interface import generate
import json

# ==========================================
# 1. Answer Prompt
# ==========================================
ANSWER_SYSTEM_PROMPT = """
# Role
You are a Senior Research Fellow with expertise in advanced quantitative sciences. Your task is to provide a "Gold Standard" solution that serves as a pedagogical reference for complex academic problems.

# Task
Execute a rigorous, step-by-step derivation for the provided problem, ensuring every logical transition is justified.

# Standardized Process
1. **Problem Analysis**: Identify the physical/mathematical framework and state all underlying assumptions.
2. **Symbolic Definition**: Explicitly define all variables, constants, and target unknowns using LaTeX.
3. **Analytical Derivation**: Construct the solution from first principles (laws, axioms, or theorems).
4. **Formal Computation**: Perform symbolic simplification or numerical evaluation with high precision.
5. **Final Synthesis**: State the final result clearly.

# Constraints
- Use LaTeX for ALL mathematical notation (e.g., $E = mc^2$).
- The final numerical or symbolic answer must be enclosed in \boxed{}.

# Problem
"""


def answer_question(question: str, model_spec: str, max_tokens=4096, temp=1.0):
    prompt = f"{ANSWER_SYSTEM_PROMPT}{question}\n\nPlease begin your complete solution:"
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=temp)


# ==========================================
# 2. Self-Correction / Attack Prompt
# ==========================================
ATTACK_SYSTEM_PROMPT = """# Role
You are a rigorous Academic Reviewer with expertise in mathematics, physics, and quantitative sciences.  
You are provided with a **Problem** and a **Candidate Solution**, which is suspected to be **INCORRECT**.

# Task
1. Begin with the assumption that the Candidate Solution contains an error.
2. Carefully examine the logical reasoning, definitions, assumptions, derivations, and calculations.
3. Identify the precise flaw or unjustified step (the error may be subtle or conceptual).
4. **Re-solve the problem from first principles**, using a clear and logically sound approach.
5. Present the corrected result clearly, and **wrap the final answer in \\boxed{}** when an explicit result is required.

# Output Format
Thinking Process: <Analyze where the error or weakness occurs>
Correct Solution: <Complete and rigorous derivation or reasoning>
Final Answer: \\boxed{<Corrected result>}

# Input
"""

def self_correct(question: str, bad_answer: str, model_spec: str, max_tokens=4096):
    prompt = (
        f"{ATTACK_SYSTEM_PROMPT}\n"
        f"### Problem\n{question}\n\n"
        f"### Suspected Incorrect Solution\n{bad_answer}\n\n"
        f"### Your Correction\n"
    )
    return generate(model_spec, prompt, max_tokens=max_tokens, temperature=0.1)