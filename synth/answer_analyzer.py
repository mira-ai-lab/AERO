import re
from collections import Counter
from local_model.local_model_interface import generate
import math

# ==========================================
# 1. Extraction Tool
# ==========================================
def extract_boxed_content(text):
    if not text:
        return None
        
    start_marker = "\\boxed{"
    start_indices = [m.start() for m in re.finditer(re.escape(start_marker), text)]
    
    if not start_indices:
        print(f"\n[Extract Failed] No '\\boxed{{' found. Text tail:\n...{text[-500:]}\n")
        return None
    
    for start_idx in reversed(start_indices):
        cursor = start_idx + len(start_marker)
        open_braces = 1 
        content_start = cursor
        
        while cursor < len(text):
            char = text[cursor]
            
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            
            if open_braces == 0:
                return text[content_start:cursor].strip()
            
            cursor += 1
            
    print(f"\n[Extract Failed] Found '\\boxed{{' but braces not closed. Text tail:\n...{text[-500:]}\n")
    return None

# ==========================================
# 2. Local Model Equivalence Check
# ==========================================
EQUIVALENCE_PROMPT = """You are a Math and Physics checker.
Determine if the following two expressions represent the same mathematical or physical value.
Consider unit conversions, scientific notation, and mathematical simplification.

Expression A: {expr_a}
Expression B: {expr_b}

Are they equivalent? 
Reply with strictly JSON: {{"equivalent": true}} or {{"equivalent": false}}.
"""

def check_equivalence_with_model(expr_a, expr_b, model_spec):
    if expr_a.replace(" ", "") == expr_b.replace(" ", ""):
        return True
    
    prompt = EQUIVALENCE_PROMPT.format(expr_a=expr_a, expr_b=expr_b)
    try:
        raw = generate(model_spec, prompt, max_tokens=1024, temperature=0.0)
        if '"equivalent": true' in raw.lower() or "'equivalent': true" in raw.lower():
            return True
        if '"equivalent": false' in raw.lower() or "'equivalent': false" in raw.lower():
            return False
        import json
        res = json.loads(raw)
        return res.get("equivalent", False)
    except:
        return False

# ==========================================
# 3. Clustering Logic
# ==========================================
def cluster_answers_with_model(answers, model_spec):
    extracted_map = [] # list of (boxed_content, original_answer)
    for ans in answers:
        content = extract_boxed_content(ans)
        if content:
            extracted_map.append({"content": content, "full_text": ans})
        else:
            extracted_map.append({"content": "PARSE_FAILED", "full_text": ans})

    # raw_groups: { "content_string": [full_text_1, full_text_2...] }
    raw_groups = {}
    for item in extracted_map:
        c = item["content"]
        if c not in raw_groups:
            raw_groups[c] = []
        raw_groups[c].append(item["full_text"])

    sorted_keys = sorted(raw_groups.keys(), key=lambda k: len(raw_groups[k]), reverse=True)
    
    # final_clusters: [ {"key": "repr_string", "count": N, "example": "full_text"} ]
    final_clusters = []
    
    for key in sorted_keys:
        if key == "PARSE_FAILED":
            continue
            
        merged = False
        for cluster in final_clusters:
            if check_equivalence_with_model(key, cluster["key"], model_spec):
                cluster["count"] += len(raw_groups[key])
                merged = True
                break
        
        if not merged:
            final_clusters.append({
                "key": key,
                "count": len(raw_groups[key]),
                "example": raw_groups[key][0]
            })

    final_clusters.sort(key=lambda x: x["count"], reverse=True)
    
    total_valid = sum(c["count"] for c in final_clusters)
    total_samples = len(answers)
    
    return {
        "clusters": final_clusters,
        "total_valid": total_valid,
        "total_samples": total_samples
    }

def analyze_distribution(clusters_data):
    """
    H_bar = - (1/log2(n)) * sum(p_i * log2(p_i)) 
    """
    clusters = clusters_data["clusters"]
    total = clusters_data["total_valid"]
    n_samples = clusters_data["total_samples"]
    
    if total == 0: return "chaos"
    
    entropy = 0.0
    for cluster in clusters:
        p_i = cluster["count"] / total
        if p_i > 0: entropy -= p_i * math.log2(p_i)
    
    norm_entropy = entropy / math.log2(n_samples) if n_samples > 1 else 0
    
    if norm_entropy < 0.3: return "mastery"
    if 0.3 <= norm_entropy <= 0.7: return "zpd"
    return "chaos"