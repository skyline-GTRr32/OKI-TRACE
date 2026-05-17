"""
Traced generation: at each step, capture logits (top-k), attention (Evidence),
and Logit Lens (prediction at each layer). Works with any Hugging Face
AutoModelForCausalLM (Llama, Qwen, Mistral, GPT-2, Phi, etc.).

Usage:
    from trace import load_model_and_tokenizer, run_traced

    model, tokenizer, _ = load_model_and_tokenizer(model_id="meta-llama/Llama-2-7b-chat-hf")
    output_text, trace = run_traced("Your prompt", max_new_tokens=64, model=model, tokenizer=tokenizer)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from entropy import (
        compute_token_entropy,
        compute_layer_entropies,
        compute_eas,
        compute_attention_entropy,
        compute_response_entropy_summary,
    )
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "model_cache")
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

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


def _format_messages_to_prompt(tokenizer, messages, enable_thinking=False):
    """
    Convert [{"role","content"}] to a string for the model.
    Uses apply_chat_template when the tokenizer has a chat_template; otherwise
    a simple "USER: ... ASSISTANT: " fallback for base models.
    When enable_thinking=True, passes it to apply_chat_template if the tokenizer
    supports it (e.g. Qwen3); ignored for tokenizers that don't.
    """
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if has_template:
        try:
            kwargs = {"messages": messages, "tokenize": False, "add_generation_prompt": True}
            if enable_thinking:
                kwargs["enable_thinking"] = True
            return tokenizer.apply_chat_template(**kwargs)
        except TypeError:
            # Tokenizer doesn't accept enable_thinking (e.g. Qwen2.5-Coder)
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


def _parse_thinking_from_output(output_text):
    """
    Parse thinking block and final answer from raw output.
    Looks for <think>...</think>
 in output; returns (thinking_text or None, final_answer).
    Works for Qwen3 native thinking and for any model that outputs such blocks.
    Ensures final_answer only contains content AFTER </think> (no thinking content).
    """
    if not output_text:
        return None, ""
    end_tag = "</think>"
    if end_tag not in output_text:
        return None, output_text.strip()
    start_tag = "<think>"
    start = output_text.find(start_tag)
    if start == -1:
        # Has </think> but no <think> - take everything after </think> as response
        end = output_text.find(end_tag)
        thinking = output_text[:end].strip()        # added the this line to fix the thinking issue with the model
        final = output_text[end + len(end_tag):].strip()
        return thinking or None, final         
    start += len(start_tag)
    end = output_text.find(end_tag, start)
    if end == -1:
        # Has <think> but no closing </think> - treat everything after <think> as thinking
        thinking = output_text[start:].strip()
        return thinking or None, ""
    thinking = output_text[start:end].strip()
    final = output_text[end + len(end_tag):].strip()
    # Ensure final doesn't contain any <think> tags (shouldn't happen but be safe)
    if "<think>" in final:
        final = final.split("<think>")[0].strip()
    return thinking or None, final


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
      Defaults to TRACE_MODEL_ID env or DeepSeek-R1-Distill-Qwen-7B.
    - use_4bit: if True, use BitsAndBytes 4-bit; on failure, fallback to fp16/bf16.
    - cache_dir: where to cache; default project model_cache.

    Returns:
        (model, tokenizer, quant_info) where quant_info is "4-bit", "bf16", or "fp16".
    """
    model_id = model_id or os.environ.get("TRACE_MODEL_ID") or DEFAULT_MODEL_ID
    cache_dir = cache_dir or CACHE_DIR

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, trust_remote_code=True
    )

    # Prefer bf16 when supported; else fp16
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    fallback_dtype_str = "bf16" if torch_dtype == torch.bfloat16 else "fp16"

    model = None
    quant_info = fallback_dtype_str
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
                trust_remote_code=True,
            )
            quant_info = "4-bit"
        except Exception:
            model = None

    if model is None:
        load_kw = {
            "cache_dir": cache_dir,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        # Prefer dtype (new) over torch_dtype (deprecated in some environments)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, **load_kw
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, **load_kw
            )
        quant_info = fallback_dtype_str

    # Prefer eager attention so output_attentions works (SDPA/Flash often don't return weights)
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
    except Exception:
        pass

    return model, tokenizer, quant_info


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
    enable_thinking: bool = False,
    compute_dla: bool = False,
    ablation_config=None,
):
    """
    Run generation and record, per step: chosen token, logits top-k, attention (Evidence),
    and Logit Lens (top-k at each layer). Works with any CausalLM.

    One of prompt or messages must be provided.
    - prompt: single user string; turned into messages=[{"role":"user","content":prompt}].
    - messages: list of {role, content}; uses apply_chat_template or a simple fallback.
    - enable_thinking: if True, use thinking mode when the tokenizer supports it (e.g. Qwen3).
      Output is parsed for think-tag blocks; thinking and final answer are stored in the trace.
    - ablation_config: optional AblationConfig from ablation.py. If provided, hooks are applied
      for the entire generation run then removed. Trace will include ablation_summary.

    Returns:
        (output_text, trace_dict)
        output_text: final answer (content after think end-tag, or full output if no thinking block).
        trace_dict: model_id, prompt, output, thinking (if any), prompt_len, messages, ...
    """
    if (prompt is None and messages is None) or (prompt is not None and messages is not None):
        raise ValueError("Provide exactly one of: prompt, messages")
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    if model is None or tokenizer is None:
        model, tokenizer, _ = load_model_and_tokenizer()

    norm, lm_head = _get_norm_and_lm_head(model)
    logit_lens_available = (norm is not None and lm_head is not None)

    device = next(model.parameters()).device
    text = _format_messages_to_prompt(tokenizer, messages, enable_thinking=enable_thinking)
    inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Forward flags: hidden_states needed for Logit Lens; attentions for Evidence
    output_attentions = True
    output_hidden_states = True
    evidence_provenance = "Attention not available (use attn_implementation='eager' or a model that returns output_attentions)"

    # --- Ablation: register hooks before generation, remove after ---
    _ablation_handles = []
    _ablation_summary = "none"
    if ablation_config is not None and not ablation_config.is_empty():
        try:
            from ablation import apply_ablation_hooks
            _ablation_handles = apply_ablation_hooks(model, ablation_config)
            _ablation_summary = ablation_config.summary()
        except Exception as e:
            _ablation_summary = f"failed: {e}"

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
        # --- Entropy signals ---
        entropy_signals = {}
        if ENTROPY_AVAILABLE:
            # Signal 1: Token Logit Entropy (full vocab, from raw logits)
            # `logits` is already computed above as: out.logits[0, -1, :].float().cpu()
            entropy_signals["token_entropy"] = compute_token_entropy(logits)

            # Signal 2 + 3: Layer Entropies and EAS (from hidden states)
            layer_ents = []
            if logit_lens_available and getattr(out, "hidden_states", None) is not None:
                layer_ents = compute_layer_entropies(
                    out.hidden_states, norm, lm_head, device
                )
                entropy_signals["layer_entropies"] = layer_ents
                entropy_signals["eas"] = compute_eas(layer_ents)
            else:
                entropy_signals["layer_entropies"] = []
                entropy_signals["eas"] = None

            # Signal 4: Attention Entropy (final layer + all-layer mean)
            if use_attn:
                entropy_signals["attention_entropy"] = compute_attention_entropy(
                    out.attentions
                )
            else:
                entropy_signals["attention_entropy"] = {
                    "final_layer": None,
                    "all_layers_mean": None,
                    "all_layers": [],
                }


        step_dict = {
            "step": step,
            "chosen_token": chosen_token,
            "chosen_id": int(next_id),
            "logits_topk": logits_topk,
            "evidence": evidence,
            "logit_lens": logit_lens,
            "entropy": entropy_signals if ENTROPY_AVAILABLE else {},
        }
        steps.append(step_dict)

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

    # --- Remove ablation hooks ---
    for h in _ablation_handles:
        h.remove()

    output_ids = input_ids[0, prompt_len:].cpu().tolist()
    output_raw = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("RAW OUTPUT:", repr(output_raw[:500]))  # added a line to see raw output 
    thinking_text, output_text = _parse_thinking_from_output(output_raw)
    print("THINKING:", repr(thinking_text))         # added a line to see raw output 


    # ---------------------------------------------------------------------------
    # DLA — analyze first meaningful output token under the same ablation that
    # was used during generation.  Three cases:
    #
    #   1. Normal run with a real response token after </think>  → "response"
    #   2. Ablated run where the model looped and never closed </think>
    #      → no response token exists, fall back to first thinking token  → "thinking"
    #   3. No thinking at all → first output token → "response"
    #
    # We re-apply ablation hooks around every DLA forward pass so the DLA
    # decomposition reflects the crippled model, not the clean one.
    # ---------------------------------------------------------------------------
    dla_first_response = None
    if compute_dla and output_ids:
        try:
            from dla_analyzer import compute_dla_for_token, find_think_end_token_index

            # --- Determine which token to analyze and label it ---
            think_end_idx = find_think_end_token_index(output_ids, tokenizer)

            # Try to find first real response token (after </think>)
            dla_token_type = "response"
            if think_end_idx is not None:
                first_response_token_index = think_end_idx + 1
                # Skip whitespace tokens
                while first_response_token_index < len(output_ids):
                    tok_str = tokenizer.decode(
                        [output_ids[first_response_token_index]], skip_special_tokens=False
                    )
                    if tok_str.strip():
                        break
                    first_response_token_index += 1

                if first_response_token_index >= len(output_ids):
                    # </think> was found but nothing came after it —
                    # model was cut off or looped without closing.
                    # Fall back to first thinking token.
                    first_response_token_index = 0
                    dla_token_type = "thinking_fallback"
            else:
                # No </think> at all: either no thinking model, or ablated
                # model that looped entirely inside thinking without ever closing.
                # Check: does the raw output look like unfinished thinking?
                if thinking_text and not output_text:
                    # Ablated loop case — analyze first thinking token
                    first_response_token_index = 0
                    dla_token_type = "thinking_fallback"
                else:
                    # Normal non-thinking model — first output token is the answer
                    first_response_token_index = 0
                    dla_token_type = "response"

            first_response_token_id = int(output_ids[first_response_token_index])
            dla_input_len = prompt_len + first_response_token_index
            dla_input_ids = input_ids[:, :dla_input_len]
            dla_attention = torch.ones_like(dla_input_ids, device=device, dtype=attention_mask.dtype)

            # --- Re-apply ablation hooks so DLA sees the ablated model ---
            _dla_handles = []
            if ablation_config is not None and not ablation_config.is_empty():
                try:
                    from ablation import apply_ablation_hooks
                    _dla_handles = apply_ablation_hooks(model, ablation_config)
                except Exception:
                    pass

            try:
                dla_first_response = compute_dla_for_token(
                    model, dla_input_ids, dla_attention, tokenizer,
                    target_token_id=first_response_token_id
                )
            finally:
                for h in _dla_handles:
                    h.remove()

            # Tag with token type so dashboard can label it correctly
            if dla_first_response and "error" not in dla_first_response:
                dla_first_response["dla_token_type"] = dla_token_type

            # --- Token attribution: which input words contributed (DLA × attention) ---
            if dla_first_response and "error" not in dla_first_response and "dla_contributions" in dla_first_response:
                try:
                    # Re-apply ablation hooks for the attention forward pass too
                    _attn_handles = []
                    if ablation_config is not None and not ablation_config.is_empty():
                        try:
                            from ablation import apply_ablation_hooks
                            _attn_handles = apply_ablation_hooks(model, ablation_config)
                        except Exception:
                            pass
                    try:
                        with torch.no_grad():
                            out_attn = model(
                                input_ids=dla_input_ids,
                                attention_mask=dla_attention,
                                output_hidden_states=False,
                                output_attentions=True,
                            )
                    finally:
                        for h in _attn_handles:
                            h.remove()

                    if getattr(out_attn, "attentions", None) is not None and len(out_attn.attentions) > 0:
                        attention_from_last = [
                            out_attn.attentions[l][0, :, -1, :].cpu().float()
                            for l in range(len(out_attn.attentions))
                        ]
                        from dla_analyzer import compute_token_attribution
                        dla_first_response["input_token_attribution"] = compute_token_attribution(
                            dla_first_response["dla_contributions"],
                            attention_from_last,
                            dla_input_ids,
                            tokenizer,
                            prompt_len=prompt_len,
                        )
                except Exception:
                    dla_first_response["input_token_attribution"] = {"error": "Could not compute (e.g. no attention)"}
        except Exception as e:
            dla_first_response = {"error": str(e)}

    model_id = getattr(model.config, "name_or_path", None) or "unknown"
    trace = {
        "model_id": model_id,
        "prompt": text,
        "output": output_text,
        "output_raw": output_raw,
        "thinking": thinking_text,
        "prompt_len": prompt_len,
        "messages": messages,
        "evidence_provenance": evidence_provenance,
        "logit_lens_available": logit_lens_available,
        "ablation_summary": _ablation_summary,
        "entropy_summary": compute_response_entropy_summary(steps) if ENTROPY_AVAILABLE else {},
        "steps": steps,
    }
    if dla_first_response is not None:
        trace["dla_first_response"] = dla_first_response
    return output_text, trace