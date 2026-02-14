"""
Traced generation: at each step, capture logits (top-k), attention (Evidence),
and Logit Lens (prediction at each layer). Works with any Hugging Face
AutoModelForCausalLM (Llama, Qwen, Mistral, GPT-2, Phi, etc.).

Usage:
    from trace import load_model_and_tokenizer, run_traced

    model, tokenizer = load_model_and_tokenizer(model_id="meta-llama/Llama-2-7b-chat-hf")
    output_text, trace = run_traced("Your prompt", max_new_tokens=64, model=model, tokenizer=tokenizer)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "model_cache")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"

# ---------------------------------------------------------------------------
# Norm + lm_head discovery (model-agnostic; architecture layouts differ)
# ---------------------------------------------------------------------------

def _get_norm_and_lm_head(model):
    """
    Find the final LayerNorm and lm_head (output embeddings) for Logit Lens.
    Tries common patterns: Llama/Qwen/Mistral (model.model.norm + lm_head),
    GPT-2 (model.transformer.ln_f + lm_head), etc. Returns (norm, lm_head);
    either can be None if not found.
    """
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "get_output_embeddings") and callable(model.get_output_embeddings):
        lm_head = model.get_output_embeddings()

    norm_candidates = [
        getattr(getattr(model, "model", None), "norm", None),
        getattr(model, "norm", None),
        getattr(getattr(model, "transformer", None), "ln_f", None),
        getattr(getattr(model, "model", None), "final_layernorm", None),
        getattr(getattr(model, "decoder", None), "final_layer_norm", None),
    ]
    for n in norm_candidates:
        if n is not None and isinstance(n, nn.Module) and lm_head is not None:
            return n, lm_head

    # Fallback: search for a final LayerNorm by name
    if lm_head is not None:
        for name, mod in model.named_modules():
            if "norm" in name.lower() and "LayerNorm" in type(mod).__name__:
                if "ln_f" in name or "final" in name or (".norm" in name and "layer" not in name.lower()):
                    return mod, lm_head
        for name, mod in model.named_modules():
            if "norm" in name.lower() and "LayerNorm" in type(mod).__name__:
                return mod, lm_head

    return (None, lm_head)


def _format_messages_to_prompt(tokenizer, messages):
    """
    Convert [{"role","content"}] to a string for the model.
    Uses apply_chat_template when the tokenizer has a chat_template; otherwise
    a simple "USER: ... ASSISTANT: " fallback for base models.
    """
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if has_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # Fallback: no chat template or apply failed (e.g. base model, old tokenizer)
    parts = []
    for m in messages:
        r = (m.get("role") or "user").upper()
        c = m.get("content") or ""
        parts.append(f"{r}: {c}")
    return "\n\n".join(parts) + "\n\nASSISTANT: "


# ---------------------------------------------------------------------------
# Model loading (model-agnostic; 4-bit optional, eager attention for output_attentions)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_id=None,
    use_4bit=True,
    cache_dir=None,
):
    """
    Load any AutoModelForCausalLM and tokenizer. Uses GPU if available.

    - model_id: HuggingFace model id (e.g. "meta-llama/Llama-2-7b-chat-hf").
      Defaults to TRACE_MODEL_ID env or Qwen2.5-Coder-3B-Instruct.
    - use_4bit: if True, use BitsAndBytes 4-bit; on failure, fallback to fp16/bf16.
    - cache_dir: where to cache; default project model_cache.
    """
    model_id = model_id or os.environ.get("TRACE_MODEL_ID") or DEFAULT_MODEL_ID
    cache_dir = cache_dir or CACHE_DIR

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # Prefer bf16 when supported; else fp16
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                quantization_config=bnb,
                device_map="auto",
            )
        except Exception:
            model = None

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch_dtype,
        )

    # Prefer eager attention so output_attentions works (SDPA/Flash often don't return weights)
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
    except Exception:
        pass

    return model, tokenizer


# ---------------------------------------------------------------------------
# Traced generation
# ---------------------------------------------------------------------------

def run_traced(
    prompt: str = None,
    messages: list = None,
    max_new_tokens: int = 64,
    model=None,
    tokenizer=None,
    topk_logits: int = 10,
    topk_evidence: int = 10,
    topk_lens: int = 5,
):
    """
    Run generation and record, per step: chosen token, logits top-k, attention (Evidence),
    and Logit Lens (top-k at each layer). Works with any CausalLM.

    One of prompt or messages must be provided.
    - prompt: single user string; turned into messages=[{"role":"user","content":prompt}].
    - messages: list of {role, content}; uses apply_chat_template or a simple fallback.

    Returns:
        (output_text, trace_dict)
        trace_dict: model_id, prompt, output, prompt_len, messages, evidence_provenance,
                    logit_lens_available, steps, ...
    """
    if (prompt is None and messages is None) or (prompt is not None and messages is not None):
        raise ValueError("Provide exactly one of: prompt, messages")
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    norm, lm_head = _get_norm_and_lm_head(model)
    logit_lens_available = (norm is not None and lm_head is not None)

    device = next(model.parameters()).device
    text = _format_messages_to_prompt(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Forward flags: hidden_states needed for Logit Lens; attentions for Evidence
    output_attentions = True
    output_hidden_states = True
    evidence_provenance = "Attention not available (use attn_implementation='eager' or a model that returns output_attentions)"

    steps = []
    for step in range(max_new_tokens):
        with torch.no_grad():
            try:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                )
            except Exception:
                # Some models fail with output_attentions or output_hidden_states
                output_attentions = False
                output_hidden_states = False
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                )

        logits = out.logits[0, -1, :].float().cpu()
        next_id = logits.argmax().item()
        chosen_token = tokenizer.decode([next_id], skip_special_tokens=False)

        # --- Logits top-k ---
        probs = F.softmax(logits, dim=-1)
        k = min(topk_logits, probs.shape[0])
        top_probs, top_ids = probs.topk(k, dim=-1)
        logits_topk = [
            {"token": tokenizer.decode([int(i)], skip_special_tokens=False), "prob": float(p)}
            for p, i in zip(top_probs.tolist(), top_ids.tolist())
        ]

        # --- Evidence (attention from last position, last layer) ---
        evidence = []
        use_attn = getattr(out, "attentions", None) is not None and len(getattr(out, "attentions", []) or []) > 0
        if use_attn:
            attn = out.attentions[-1][0, :, -1, :].mean(0).cpu()
            k_ev = min(topk_evidence, attn.shape[0])
            top_weights, top_pos = attn.topk(k_ev, dim=-1)
            ids_at_pos = input_ids[0].cpu()
            for w, pos in zip(top_weights.tolist(), top_pos.tolist()):
                pos = int(pos)
                if pos < ids_at_pos.shape[0]:
                    t = tokenizer.decode([ids_at_pos[pos].item()], skip_special_tokens=False)
                    in_prompt = pos < prompt_len
                    relative = (pos - prompt_len) if pos >= prompt_len else None
                    evidence.append({"token": t, "weight": round(w, 5), "position": pos, "in_prompt": in_prompt, "relative": relative})
        if use_attn:
            evidence_provenance = "Attention from final layer (heads averaged)"

        # --- Logit Lens ---
        logit_lens = []
        if logit_lens_available and getattr(out, "hidden_states", None) is not None:
            for layer in range(len(out.hidden_states)):
                h = out.hidden_states[layer][0, -1, :]
                with torch.no_grad():
                    h_norm = norm(h.unsqueeze(0))
                    logits_l = lm_head(h_norm)[0].float().cpu()
                probs_l = F.softmax(logits_l, dim=-1)
                k_l = min(topk_lens, probs_l.shape[0])
                top_p, top_i = probs_l.topk(k_l, dim=-1)
                logit_lens.append({
                    "layer": layer,
                    "topk": [
                        {"token": tokenizer.decode([int(i)], skip_special_tokens=False), "prob": round(float(p), 4)}
                        for p, i in zip(top_p.tolist(), top_i.tolist())
                    ],
                })

        steps.append({
            "step": step,
            "chosen_token": chosen_token,
            "chosen_id": int(next_id),
            "logits_topk": logits_topk,
            "evidence": evidence,
            "logit_lens": logit_lens,
        })

        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)],
            dim=1,
        )
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
            dim=1,
        )
        if eos_id is not None and next_id == eos_id:
            break

    output_ids = input_ids[0, prompt_len:].cpu().tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    model_id = getattr(model.config, "name_or_path", None) or "unknown"
    trace = {
        "model_id": model_id,
        "prompt": text,
        "output": output_text,
        "prompt_len": prompt_len,
        "messages": messages,
        "evidence_provenance": evidence_provenance,
        "logit_lens_available": logit_lens_available,
        "steps": steps,
    }
    return output_text, trace
