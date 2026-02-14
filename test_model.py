"""
Test that the model loads, runs on GPU, and produces output.

Run from project root (with venv activated):
    python test_model.py

Uses 4-bit quantization to fit in 6GB VRAM. Requires: model_cache/ populated
(via python download_model.py).
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Match download_model.py
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "model_cache")
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"


def main():
    print("Checking GPU...")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("  No GPU found. Using CPU (slower).")
        device = "cpu"

    print(f"Loading tokenizer from cache: {CACHE_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    print("Loading model (4-bit for 6GB VRAM)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        quantization_config=bnb,
        device_map="auto",
    )

    prompt = "Write a one-line Python function that adds two numbers."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    print(f"Prompt: {prompt}")
    print("Generating...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the new tokens (skip the prompt part)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("-" * 40)
    print("Output:")
    print(response[:500] + ("..." if len(response) > 500 else ""))
    print("-" * 40)
    print("Model runs and gives output. Test OK.")


if __name__ == "__main__":
    main()
