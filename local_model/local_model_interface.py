# local_model/local_model_interface.py
"""
支持三类本地模型访问方式：
- hf::<model_name> -> 使用 transformers AutoModelForCausalLM + tokenizer
- local::<path>   -> 假设 path 是 transformers checkpoint dir
- http::<url>     -> 发送 POST 到本地推理服务（JSON { "prompt": "..."}）
接口：generate(prompt, max_tokens=512, temperature=0.2)
"""
import os
import time
from typing import Optional
import requests

# Lazy import transformers only if used
_transformers_loaded = False
_tokenizer = None
_model = None

def _load_transformers(model_spec: str):
    global _transformers_loaded, _tokenizer, _model
    if _transformers_loaded:
        return
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    # model_spec might be "hf::gpt2" or "local::/path"
    tag = model_spec.split("::", 1)[0]
    spec = model_spec.split("::", 1)[1]
    print(f"[local_model] loading transformers model: {spec}")
    _tokenizer = AutoTokenizer.from_pretrained(spec, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(spec, device_map="auto")
    _transformers_loaded = True

def generate_from_transformers(model_spec: str, prompt: str, max_tokens=512, temperature=0.2):
    _load_transformers(model_spec)
    input_ids = _tokenizer(prompt, return_tensors="pt").input_ids.to(_model.device)
    gen = _model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=float(temperature),
        pad_token_id=_tokenizer.eos_token_id
    )
    out = _tokenizer.decode(gen[0], skip_special_tokens=True)
    # Remove the prompt prefix if model echoes it
    if out.startswith(prompt):
        return out[len(prompt):].strip()
    return out.strip()

def generate_from_http(url: str, prompt: str, max_tokens=2048, temperature=0.2):
    # url 应该是 vLLM 的基础 URL，例如 http://localhost:8000
    # 附加 /v1/chat/completions
    
    # 确保 URL 基础正确
    if url.endswith("/generate"):
        base_url = url.replace("/generate", "")
    else:
        base_url = url
        
    completion_url = f"{base_url.rstrip('/')}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "psp_model", 
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    try:
        resp = requests.post(completion_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # 解析 OpenAI 格式的响应
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling vLLM API at {completion_url}: {e}")
        # 返回空字符串或重新引发异常，以匹配原始逻辑
        raise e

def generate(model_spec: str, prompt: str, max_tokens=512, temperature=0.2, enable_thinking=False):
    if model_spec.startswith("hf::") or model_spec.startswith("local::"):
        return generate_from_transformers(model_spec, prompt, max_tokens, temperature)
    elif model_spec.startswith("http::"):
        # 1. 提取 URL 列表部分
        raw_urls = model_spec.split("::", 1)[1]
        # 2. 支持逗号分隔的多个 URL: "http://host:8001,http://host:8002"
        url_list = raw_urls.split(",")
        # 3. 随机选择一个后端进行负载均衡
        target_url = random.choice(url_list)
        return generate_from_http(target_url, prompt, max_tokens, temperature, enable_thinking)
    else:
        raise ValueError("Unsupported model_spec format.")
