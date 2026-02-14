"""
Download any Hugging Face CausalLM and cache it in the project's model_cache.

Run from project root:
    python download_model.py [MODEL_ID]
    python download_model.py                          # uses default Qwen2.5-Coder-3B
    TRACE_MODEL_ID=meta-llama/Llama-2-7b-chat-hf python download_model.py

Uses resume_download; if the download fails, run again to resume.
Cached under: <project_root>/model_cache/
"""

import os
import sys
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "model_cache")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"

def main():
    model_id = os.environ.get("TRACE_MODEL_ID") or (sys.argv[1] if len(sys.argv) > 1 else None) or DEFAULT_MODEL_ID
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache directory: {os.path.abspath(CACHE_DIR)}")
    print(f"Downloading {model_id} ...")
    snapshot_download(
        model_id,
        cache_dir=CACHE_DIR,
        resume_download=True,
        max_workers=1,
    )
    print(f"Done. Cached under: {CACHE_DIR}")

if __name__ == "__main__":
    main()
