# aggregate_results.py
import glob
import os
import json
from typing import List, Dict
import argparse 

# --- 因为这个脚本在根目录，我们直接从 utils.io 导入 ---
# (确保你从根目录运行这个脚本)
try:
    from utils.io import read_jsonl, write_jsonl
except ImportError:
    print("Import Error: 确保你在 psp/ 根目录下运行此脚本。")
    print("正在尝试备用导入...")
    
    # --- Fallback: 如果直接导入失败，定义在本地 ---
    # (这段代码与 utils/io.py 相同)
    def read_jsonl(path: str) -> List[Dict]:
        out = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out

    def write_jsonl(path: str, records: List[Dict]):
        with open(path, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

def aggregate_verified_qa(base_dir="outputs", output_file="master_qa_dataset.jsonl"):
    """
    遍历所有轮次的 outputs/ 目录，提取所有被验证为“通过”的
    {question, final_answer} 对。
    """
    all_verified_pairs = []
    # 搜索所有轮次的 inner_results.jsonl 文件
    search_path = os.path.join(base_dir, "round_*", "inner_results.jsonl")
    result_files = glob.glob(search_path)
    
    if not result_files:
        print(f"未在 {search_path} 找到任何 inner_results.jsonl 文件。")
        return

    print(f"找到了 {len(result_files)} 个轮次的结果文件，正在处理...")
    
    total_records = 0
    
    for f_path in sorted(result_files): # 按轮次排序
        print(f"  正在读取 {f_path}...")
        try:
            records = read_jsonl(f_path)
            for rec in records:
                status = rec.get("status", "")
                
                # [关键] 检查 inner_loop.py 中定义的“通过”状态
                if status == "passed_initial" or status.startswith("refined_pass"):
                    pair = {
                        "question": rec.get("question"),
                        "answer": rec.get("final_answer"), #
                        "source_round": f_path # (可选) 记录来源
                    }
                    all_verified_pairs.append(pair)
            
            total_records += len(records)
            
        except Exception as e:
            print(f"读取或处理 {f_path} 出错: {e}")
            
    print("\n--- 聚合完成 ---")
    print(f"总共处理了 {total_records} 条记录。")
    print(f"提取了 {len(all_verified_pairs)} 条已验证的“正确”问答对。")
    
    if all_verified_pairs:
        write_jsonl(output_file, all_verified_pairs)
        print(f"✅ 成功保存到 {output_file}")
    else:
        print("未提取到任何数据，未生成文件。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None, help="实验名称")
    args = parser.parse_args()

    if args.exp_name:
        base_dir = os.path.join("experiments", args.exp_name, "outputs")
        output_file = os.path.join("experiments", args.exp_name, "master_qa_dataset.jsonl")
    else:
        base_dir = "outputs"
        output_file = "master_qa_dataset.jsonl"
        
    aggregate_verified_qa(base_dir=base_dir, output_file=output_file)

# 脚本执行完毕后，你将在根目录下找到一个 master_qa_dataset.jsonl 文件，其中包含了所有轮次中被验证为正确的 {question, answer} 数据集。