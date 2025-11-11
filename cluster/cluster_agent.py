# cluster/cluster_agent.py
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import entropy
import json
from utils.io import read_text, write_text

class ClusterAgent:
    def __init__(self, model_name="all-MiniLM-L6-v2", n_clusters=8):
        self.embed_model = SentenceTransformer(model_name)
        self.n_clusters = n_clusters

    def cluster_questions(self, questions):
        if not questions:
            return {"labels": [], "counts": [], "probs": [], "entropy": 0.0}
        embs = self.embed_model.encode(questions, convert_to_numpy=True)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(embs)
        labels = kmeans.labels_
        counts = np.bincount(labels, minlength=self.n_clusters)
        probs = counts / counts.sum()
        ent = float(entropy(probs))
        return {"labels": labels.tolist(), "counts": counts.tolist(), "probs": probs.tolist(), "entropy": ent}

    def analyze_and_suggest(self, questions, entropy_threshold=1.2):
        res = self.cluster_questions(questions)
        ent = res["entropy"]
        suggestion = None
        if ent < entropy_threshold:
            counts = np.array(res["counts"])
            small_idx = np.argsort(counts)[:max(1, len(counts)//4)].tolist()
            suggestion = {
                "reason": "low_entropy",
                "entropy": ent,
                "small_clusters": small_idx,
                "prompt_suggestion": f"请生成与现有簇不同、着重电磁/热学/多步推导题型，优先覆盖簇索引 {small_idx}。"
            }
        return {"cluster_info": res, "suggestion": suggestion}

    def apply_suggestion_to_prompt(self, suggestion, prompt_path="synth/prompt_template.txt"):
        if not suggestion:
            return
        text = read_text(prompt_path)
        # 简单替换 focus 字段
        new_focus = "电磁/热学/多步推导"
        new_text = text.replace("{focus}", new_focus)
        write_text(prompt_path, new_text)
        return new_text
