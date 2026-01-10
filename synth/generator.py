# synth/generator.py
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from local_model.local_model_interface import generate
from utils.io import read_text
from tqdm import tqdm

PROMPT_FILE = "synth/prompt_template.txt"

# # [新增] 定义丰富且细分的物理领域列表
# PHYSICS_TOPICS = [
#     "Atomic Physics (Bohr model, hydrogen atom, energy levels, spectral lines, fine structure)",
#     "Classical Electromagnetism (Electrostatics, Gauss's Law, magnetostatics, Ampere's Law, Maxwell's equations)",
#     "Classical Mechanics (Newton's laws, kinematics, dynamics, work and energy, momentum, collisions)",
#     "Electrodynamics (Electromagnetic waves, radiation, special relativity in electrodynamics, potentials)",
#     "Geometrical Optics (Reflection, refraction, lenses, mirrors, optical instruments, ray tracing)",
#     "Quantum Mechanics (Schrödinger equation, wave functions, operators, uncertainty principle, harmonic oscillator)",
#     "Relativity (Special relativity, time dilation, length contraction, Lorentz transformations, energy-momentum relation)",
#     "Semiconductor Physics (Band theory, pn junctions, transistors, carrier transport, doping)",
#     "Solid-State Physics (Crystal structure, lattice vibrations, thermal properties, electronic properties of solids)",
#     "Statistical Mechanics (Ensembles, partition functions, Boltzmann distribution, thermodynamic probabilities, quantum statistics)",
#     "Theoretical Mechanics (Lagrangian mechanics, Hamiltonian mechanics, generalized coordinates, principle of least action)",
#     "Thermodynamics (Laws of thermodynamics, entropy, heat engines, phase transitions, thermodynamic potentials)",
#     "Wave Optics (Interference, diffraction, polarization, coherence, double-slit experiment)"
# ]
PHYSICS_TOPICS = [
    "Classical Mechanics",
    "Electromagnetism",
    "Thermodynamics",
    "Statistical Mechanics",
    "Quantum Mechanics",
    "Relativity",
    "Wave and Optics",
    "Solid State Physics"
]

def load_prompt_template():
    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"{PROMPT_FILE} not found.")
    return read_text(PROMPT_FILE)

# max_workers 默认为 30，可根据显存情况调整
def generate_questions(n: int, model_spec: str, temperature=1.0, max_tokens=1024, max_workers=30):
    raw_template = load_prompt_template()
    questions = []

    def _generate_single(idx):
        prompt_text = raw_template  # 直接使用模板

        try:
            out = generate(model_spec, prompt_text, max_tokens=max_tokens, temperature=temperature)
            data = json.loads(out.strip())
            return {
                "question": data.get("question", out), 
                "meta": data.get("meta", {}),
                "prompt": prompt_text
            }
        except Exception:
            # 出错时返回原始输出或空字符串
            return {
                "question": out.strip() if 'out' in locals() else "", 
                "meta": {},
                "prompt": prompt_text
            }
    # def _generate_single(idx):
    #     prompt_text = raw_template 

    #     try:
    #         # 直接获取模型生成的文本
    #         out = generate(model_spec, prompt_text, max_tokens=max_tokens, temperature=temperature)
            
    #         # --- 核心简化：不再尝试解析 JSON ---
    #         question_content = out.strip()
            
    #         # 如果模型不听话加了 ```markdown 块，可以简单清理（可选）
    #         if question_content.startswith("```"):
    #             # 移除开头的 ```... 和结尾的 ```
    #             question_content = re.sub(r'^```[\w]*\n', '', question_content)
    #             question_content = re.sub(r'\n```$', '', question_content)

    #         return {
    #             "question": question_content, 
    #             "meta": {}, # 由于不再要求 JSON，这里置空或存储默认值
    #             "prompt": prompt_text
    #         }
    #     except Exception as e:
    #         print(f"Error during single generation: {e}")
    #         return {
    #             "question": "", 
    #             "meta": {},
    #             "prompt": prompt_text
    #         }

    print(f"Generating {n} questions with {max_workers} workers")
    
    # 使用线程池并发请求
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交 n 个任务，传入 i 作为索引以实现轮询
        futures = [executor.submit(_generate_single, i) for i in range(n)]
        
        # 使用 tqdm 包装 as_completed，显示生成进度
        for future in tqdm(as_completed(futures), total=n, desc="Generating Questions"):
            try:
                res = future.result()
                if res["question"]:
                    questions.append(res)
            except Exception as e:
                print(f"Generation task failed: {e}")

    return questions