# utils/make_dpo_pairs.py
import os, json
from utils.io import read_jsonl

def convert_pairs_to_sharegpt(infile, outfile):
    recs = read_jsonl(infile)
    out = []
    for r in recs:
        # 兼容三种数据的 prompt 字段
        # 1. Answer Pairs: key="question"
        # 2. Question Pairs: key="prompt" (inner_loop 可能没写这个key, 但我们用 chosen 做 prompt?) 
        #    不，Questions DPO 是 Generator 训练，Prompt 通常是 "请基于...生成题目"
        # 3. Critic Pairs: key="prompt" (这是完整的 Critic 指令)
        
        user_input = r.get("question") or r.get("prompt") or ""
        
        # 对于 Question DPO，chosen/rejected 是题目本身，user_input 应该是 prompt_template 的内容
        # 这里的逻辑需要根据 inner_loop 的输出稍微注意一下
        # 但针对 Critic Pairs，r["prompt"] 就是完整的输入，r["chosen"] 是 JSON 字符串
        
        out.append({
            "conversations": [{"from": "human", "value": user_input}],
            "chosen": {"from": "gpt", "value": r["chosen"]},
            "rejected": {"from": "gpt", "value": r["rejected"]}
        })
    
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(out)} records to ShareGPT format in {outfile}")

if __name__ == "__main__":
    os.makedirs("dpo_data", exist_ok=True)
    
    if os.path.exists("dpo_data/answers_pairs.jsonl"):
        convert_pairs_to_sharegpt("dpo_data/answers_pairs.jsonl", "dpo_data/answers_dpo.json")
    
    if os.path.exists("dpo_data/questions_pairs.jsonl"):
        # 注意：Question DPO 的 prompt 在 inner_loop 里并没有很好地保存，
        # 之前的代码 questions_pairs 只有 chosen/rejected。
        # 建议在 inner_loop 的 questions_pairs 里补上 "prompt": "请生成..."
        convert_pairs_to_sharegpt("dpo_data/questions_pairs.jsonl", "dpo_data/questions_dpo.json")

    # [新增] 处理 Critic 数据
    if os.path.exists("dpo_data/critic_pairs.jsonl"):
        convert_pairs_to_sharegpt("dpo_data/critic_pairs.jsonl", "dpo_data/critic_dpo.json")