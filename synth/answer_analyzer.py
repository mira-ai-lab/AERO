# synth/answer_analyzer.py
import re
from collections import Counter
from local_model.local_model_interface import generate

# ==========================================
# 1. Extraction Tool
# ==========================================
def extract_boxed_content(text):
    """
    提取 \\boxed{} 中的内容，支持嵌套的大括号（如 \\boxed{\\frac{1}{2}}）。
    如果有多个 \\boxed，取最后一个。
    """
    if not text:
        return None
        
    # 找到所有 "\boxed{" 的起始位置
    start_marker = "\\boxed{"
    start_indices = [m.start() for m in re.finditer(re.escape(start_marker), text)]
    
    # [Case 1] 根本没有找到 \boxed
    if not start_indices:
        print(f"\n[Extract Failed] No '\\boxed{{' found. Text tail:\n...{text[-500:]}\n")
        return None
    
    # 我们只需要最后一个 boxed，但为了稳健，可以从后往前找，找到第一个合法的就返回
    for start_idx in reversed(start_indices):
        # 指针移动到 "boxed{" 之后的内容开始处
        cursor = start_idx + len(start_marker)
        open_braces = 1 # 已经消耗了 \boxed 的那个 {
        content_start = cursor
        
        # 开始向后扫描
        while cursor < len(text):
            char = text[cursor]
            
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            
            # 如果括号计数归零，说明找到了匹配的右括号
            if open_braces == 0:
                return text[content_start:cursor].strip()
            
            cursor += 1
            
    # [Case 2] 找到了 \boxed 但括号没有正确闭合（通常是因为 max_tokens 截断）
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
    """
    调用本地模型判断两个表达式是否等价
    """
    # 简单的字符串预处理
    if expr_a.replace(" ", "") == expr_b.replace(" ", ""):
        return True
    
    prompt = EQUIVALENCE_PROMPT.format(expr_a=expr_a, expr_b=expr_b)
    try:
        # max_tokens 很小即可，温度为 0
        raw = generate(model_spec, prompt, max_tokens=1024, temperature=0.0)
        # 简单的解析，防止模型废话
        if '"equivalent": true' in raw.lower() or "'equivalent': true" in raw.lower():
            return True
        if '"equivalent": false' in raw.lower() or "'equivalent': false" in raw.lower():
            return False
        # Fallback parsing
        import json
        res = json.loads(raw)
        return res.get("equivalent", False)
    except:
        return False

# ==========================================
# 3. Clustering Logic
# ==========================================
def cluster_answers_with_model(answers, model_spec):
    """
    对 N 个回答进行聚类。
    步骤：
    1. 提取 boxed 内容。
    2. 基于字符串的粗聚类 (Raw Grouping)。
    3. 基于模型的合并 (Model Merging)：判断不同 Raw Group 是否其实是同一个答案。
    """
    # 1. 提取
    extracted_map = [] # list of (boxed_content, original_answer)
    for ans in answers:
        content = extract_boxed_content(ans)
        if content:
            extracted_map.append({"content": content, "full_text": ans})
        else:
            extracted_map.append({"content": "PARSE_FAILED", "full_text": ans})

    # 2. 粗聚类 (String Exact Match)
    # raw_groups: { "content_string": [full_text_1, full_text_2...] }
    raw_groups = {}
    for item in extracted_map:
        c = item["content"]
        if c not in raw_groups:
            raw_groups[c] = []
        raw_groups[c].append(item["full_text"])

    # 排序：按频率从高到低
    sorted_keys = sorted(raw_groups.keys(), key=lambda k: len(raw_groups[k]), reverse=True)
    
    # 3. 模型合并 (Merge equivalent keys)
    # final_clusters: [ {"key": "repr_string", "count": N, "example": "full_text"} ]
    final_clusters = []
    
    # 这里的逻辑是：拿出一个新 key，跟已有的 cluster key 比较，如果等价则合并，否则新建 cluster
    # 忽略 PARSE_FAILED
    for key in sorted_keys:
        if key == "PARSE_FAILED":
            continue
            
        merged = False
        for cluster in final_clusters:
            # 比较 key 和 cluster 的代表 key
            if check_equivalence_with_model(key, cluster["key"], model_spec):
                cluster["count"] += len(raw_groups[key])
                # 我们可以保留原来的 example，或者 update
                merged = True
                break
        
        if not merged:
            final_clusters.append({
                "key": key,
                "count": len(raw_groups[key]),
                "example": raw_groups[key][0] # 存一个范例用于后续 Self-Correction
            })

    # 重新按 count 排序
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
    [修改版] 简化逻辑：只看 Top 1 的占比。
    如果 Top 1 占比在 0.3 到 0.8 之间，认为是好问题 (Bimodal)。
    """
    clusters = clusters_data["clusters"]
    total = clusters_data["total_valid"] 
    
    # 异常情况：没有提取到任何有效答案
    if total == 0:
        return "chaotic"
    
    top1_ratio = clusters[0]["count"] / total
    top2_ratio = clusters[1]["count"] / total
    
    if top1_ratio > 0.9 :
        return "consistent"
    
    if top1_ratio < 0.2:
        return "chaotic"

    if top1_ratio >= 0.3 and top1_ratio <= 0.6 and top1_ratio + top2_ratio >= 0.6:
        return "bimodal"
            
    # Case C: Chaotic (太难 / 发散)
    # Top 1 占比低于 30%，说明模型完全在乱猜，没有形成共识
    return "drop"