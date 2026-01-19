# local_model/local_model_interface.py
import os
import time
from typing import Optional
import requests
import random

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
    if out.startswith(prompt):
        return out[len(prompt):].strip()
    return out.strip()

def generate_from_http(url: str, prompt: str, max_tokens=2048, temperature=0.2):
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
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling vLLM API at {completion_url}: {e}")
        raise e

def generate(model_spec: str, prompt: str, max_tokens=512, temperature=0.2, enable_thinking=False):
    if model_spec.startswith("hf::") or model_spec.startswith("local::"):
        return generate_from_transformers(model_spec, prompt, max_tokens, temperature)
    elif model_spec.startswith("http::"):
        raw_urls = model_spec.split("::", 1)[1]
        # URL: "http://host:8001,http://host:8002"
        url_list = raw_urls.split(",")
        target_url = random.choice(url_list)
        return generate_from_http(target_url, prompt, max_tokens, temperature)
    else:
        raise ValueError("Unsupported model_spec format.")
