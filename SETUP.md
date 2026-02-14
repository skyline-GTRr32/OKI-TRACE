# Setup

## 1. Create a Python virtual environment

**Windows (PowerShell or CMD):**
```powershell
cd c:\Users\ALI\Desktop\os
python -m venv venv
.\venv\Scripts\activate
```

**Windows (Git Bash):**
```bash
cd /c/Users/ALI/Desktop/os
python -m venv venv
source venv/Scripts/activate
```

**macOS / Linux:**
```bash
cd /path/to/os
python3 -m venv venv
source venv/bin/activate
```

When the venv is active, the prompt usually starts with `(venv)`.

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 2b. Check CUDA / GPU (optional)

```powershell
python check_cuda.py
```

If it says `CUDA available: False`, reinstall PyTorch with CUDA: `pip uninstall torch` then `pip install torch --index-url https://download.pytorch.org/whl/cu121`. See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

## 3. Download a model (cached in `model_cache/`)

**Default (Qwen2.5-Coder-3B-Instruct):**
```powershell
python download_model.py
```

**Any HuggingFace CausalLM:**
```powershell
python download_model.py meta-llama/Llama-2-7b-chat-hf
# or
$env:TRACE_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"; python download_model.py
```

The dashboard and tracer will also download on first use if the model is not cached.

**Test the model (load, run on GPU, print output):**
```powershell
python test_model.py
```

**v1 Trace — test tracer and save a sample:**
```powershell
python test_trace.py
```
Prints a summary and step 0, and writes `trace_sample.json`.

**v1 Dashboard (Streamlit):**
```powershell
streamlit run dashboard.py
```
Set **Model ID** in the sidebar (e.g. `Qwen/Qwen2.5-Coder-3B-Instruct`, `meta-llama/Llama-2-7b-chat-hf`). Chat, then use **View trace** on any assistant reply (or load `trace_sample.json`). Use **Reload model** after changing Model ID or toggling 4-bit.

**If the download fails** (e.g. `IncompleteRead` or `ChunkedEncodingError`): run `python download_model.py` again; it will resume. For a smaller/faster model: `python download_model.py Qwen/Qwen2.5-0.5B-Instruct`. To load in code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    cache_dir="./model_cache",
    # ... e.g. device_map="auto", quantization_config=...
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    cache_dir="./model_cache",
)
```

## 4. Deactivate the venv when done

```powershell
deactivate
```
