"""
Direct Logit Attribution (DLA) for Qwen/DeepSeek-style transformers.

Decomposes the model's logits into exact contributions from each component
(embedding + per-layer attention and MLP) using fixed RMS from the forward pass.
Uses torch.no_grad() throughout. Designed for batch_size=1, last token only.

Set DLA_DEBUG=1 in the environment to print shapes and sum vs actual for diagnosis.
"""

from __future__ import annotations

import os
import torch
from typing import Any, Dict, List, Optional, Tuple

DLA_DEBUG = os.environ.get("DLA_DEBUG", "").strip() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Hook setup: capture component outputs at last token position [0, -1, :]
# ---------------------------------------------------------------------------

def setup_dla_hooks(model: torch.nn.Module) -> Tuple[List[Any], Dict[str, torch.Tensor]]:
    """
    Register forward hooks to capture component outputs at last position only.
    Returns (list of hook handles to remove later, storage dict with captured tensors).
    Assumes Qwen/DeepSeek layout: model.model.layers[i].self_attn, .mlp, model.model.norm.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        inner = getattr(model, "transformer", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("Model has no model.model.layers (not a Qwen/DeepSeek layout?)")

    storage: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []

    # --- Embedding: input to first layer at last position (pre_hook: module, input only) ---
    def _capture_embed(module: torch.nn.Module, input: Tuple[torch.Tensor, ...]) -> None:
        x = input[0]
        if x.dim() == 3:
            storage["embed"] = x[0, -1, :].detach().clone().float()

    h = layers[0].register_forward_pre_hook(_capture_embed)
    handles.append(h)

    # --- Per-layer: attention output then MLP output (before residual add) ---
    n_layers = len(layers)
    for i in range(n_layers):
        # Attention output: self_attn returns (attn_output, ...)
        def _make_attn_capture(idx: int):
            def _capture(module: torch.nn.Module, input: Tuple, output: Any) -> None:
                out = output[0] if isinstance(output, (tuple, list)) else output
                if out.dim() == 3:
                    storage[f"L{idx}_attn"] = out[0, -1, :].detach().clone().float()
            return _capture

        h_attn = layers[i].self_attn.register_forward_hook(_make_attn_capture(i))
        handles.append(h_attn)

        # MLP output
        def _make_mlp_capture(idx: int):
            def _capture(module: torch.nn.Module, input: Tuple, output: torch.Tensor) -> None:
                if output.dim() == 3:
                    storage[f"L{idx}_mlp"] = output[0, -1, :].detach().clone().float()
            return _capture

        h_mlp = layers[i].mlp.register_forward_hook(_make_mlp_capture(i))
        handles.append(h_mlp)

    # --- Final residual: output of last layer at last position ---
    def _capture_final(module: torch.nn.Module, input: Tuple, output: Any) -> None:
        out = output[0] if isinstance(output, (tuple, list)) else output
        if out.dim() == 3:
            storage["final_residual"] = out[0, -1, :].detach().clone().float()

    h_final = layers[-1].register_forward_hook(_capture_final)
    handles.append(h_final)

    return handles, storage


def _get_norm_lm_head_eps(model: torch.nn.Module) -> Tuple[torch.nn.Module, torch.nn.Module, float]:
    """Get final RMSNorm module, lm_head, and eps (variance_epsilon)."""
    inner = getattr(model, "model", None) or getattr(model, "transformer", model)
    norm = getattr(inner, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "get_output_embeddings") and callable(model.get_output_embeddings):
        lm_head = model.get_output_embeddings()
    if norm is None or lm_head is None:
        raise ValueError("Could not find model.model.norm or model.lm_head")
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    return norm, lm_head, eps


# ---------------------------------------------------------------------------
# DLA computation: fixed RMS*, normalize, unembed
# ---------------------------------------------------------------------------

def compute_dla_for_token(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    tokenizer: Any,
    target_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run one forward pass with hooks, then compute DLA for the chosen token.
    input_ids: [1, seq_len] (batch_size=1). Target is the next token (argmax of logits at last pos).
    If target_token_id is None, uses argmax of actual logits at last position.
    Returns dict with dla_contributions, dla_summary, verification.
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "Requires batch_size=1"

    handles, storage = setup_dla_hooks(model)
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
            )
    finally:
        for h in handles:
            h.remove()

    logits = out.logits
    actual_logits = logits[0, -1, :].float()
    if target_token_id is None:
        target_token_id = int(actual_logits.argmax().item())
    chosen_token = tokenizer.decode([target_token_id], skip_special_tokens=False)

    # Build ordered list of (name, component_tensor)
    component_names: List[str] = ["embed"]
    inner = getattr(model, "model", None) or getattr(model, "transformer", model)
    n_layers = len(getattr(inner, "layers", []))
    for i in range(n_layers):
        component_names.append(f"L{i}_attn")
        component_names.append(f"L{i}_mlp")

    if "final_residual" not in storage:
        return {
            "error": "final_residual not captured (hooks may not have fired)",
            "chosen_token": chosen_token,
            "chosen_token_id": target_token_id,
        }

    norm_module, lm_head_module, eps = _get_norm_lm_head_eps(model)
    gamma = norm_module.weight.detach().float()
    W_U = lm_head_module.weight.detach().float()
    final_residual = storage["final_residual"]
    d_model = final_residual.numel()

    # RMS* = sqrt(eps + (1/d) * sum(x^2))
    RMS_star = torch.sqrt(eps + (1.0 / d_model) * torch.sum(final_residual ** 2))
    RMS_star = RMS_star.item() if RMS_star.dim() == 0 else RMS_star

    # Device for computation (match model)
    device = next(model.parameters()).device
    gamma = gamma.to(device)
    W_U = W_U.to(device)

    contributions_logit_vectors: List[torch.Tensor] = []
    contributions_to_token: Dict[str, float] = {}

    for name in component_names:
        if name not in storage:
            continue
        comp = storage[name].to(device)
        # normalized = (gamma * component) / RMS_star
        normalized = (gamma * comp) / RMS_star
        # logit_contribution = normalized @ W_U.T  -> [vocab_size]
        logit_vec = normalized @ W_U.T
        contributions_logit_vectors.append(logit_vec)
        contributions_to_token[name] = logit_vec[target_token_id].item()

    # Relaxed tolerance for 4-bit/8-bit quantized models (numerical instability)
    is_quantized = (
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or (getattr(model, "quantization_config", None) is not None)
    )
    atol = 0.15 if is_quantized else 1e-3

    # Sum of all component logits must equal actual logits (exact decomposition in fp16; approximate in 4-bit)
    sum_logits = sum(contributions_logit_vectors)
    actual_at_last = actual_logits.to(sum_logits.device)
    error = (sum_logits - actual_at_last).abs().max().item()
    passed = torch.allclose(sum_logits, actual_at_last, atol=atol)

    if DLA_DEBUG:
        n_components = len(contributions_to_token)
        embed_contrib = contributions_to_token.get("embed", 0.0)
        sum_attn = sum(contributions_to_token.get(f"L{i}_attn", 0) for i in range(n_layers))
        sum_mlp = sum(contributions_to_token.get(f"L{i}_mlp", 0) for i in range(n_layers))
        sum_all = sum(contributions_to_token.values())
        actual_chosen = actual_logits[target_token_id].item()
        print(f"[DLA_DEBUG] Number of components: {n_components}")
        print(f"[DLA_DEBUG] final_residual.shape: {final_residual.shape} (d_model={d_model})")
        for name in list(storage.keys())[:5]:
            t = storage[name]
            print(f"[DLA_DEBUG]   {name}.shape: {t.shape}")
        if len(storage) > 5:
            print(f"[DLA_DEBUG]   ... and {len(storage) - 5} more")
        print(f"[DLA_DEBUG] Embed contrib (to chosen token): {embed_contrib:.4f}")
        print(f"[DLA_DEBUG] Total attn contrib: {sum_attn:.4f}")
        print(f"[DLA_DEBUG] Total mlp contrib: {sum_mlp:.4f}")
        print(f"[DLA_DEBUG] Sum of all (to chosen token): {sum_all:.4f}")
        print(f"[DLA_DEBUG] Actual logit (chosen token): {actual_chosen:.4f}")
        print(f"[DLA_DEBUG] Difference: {sum_all - actual_chosen:.4f}")
        print(f"[DLA_DEBUG] Max error over full vocab: {error:.6f}")
        print(f"[DLA_DEBUG] atol={atol} (quantized={is_quantized}), passed={passed}")

    if not passed:
        # Print debug info once when verification fails
        if not DLA_DEBUG:
            n_components = len(contributions_to_token)
            sum_all = sum(contributions_to_token.values())
            actual_chosen = actual_logits[target_token_id].item()
            print(f"[DLA] Verification failed. Components={n_components}, sum(contrib)={sum_all:.4f}, actual={actual_chosen:.4f}, max_error={error:.6f}. Set DLA_DEBUG=1 for full dump.")
        raise AssertionError(
            f"DLA verification failed: max |sum - actual| = {error:.6f} (atol={atol}). "
            "Use float16 for exact DLA, or atol is relaxed for 4-bit models."
        )

    # Percentages: 100 * |contribution| / sum(|contributions|) for chosen token, or by total effect
    total_abs = sum(abs(contributions_to_token[k]) for k in contributions_to_token)
    if total_abs < 1e-9:
        total_abs = 1.0
    dla_contributions: Dict[str, Dict[str, Any]] = {}
    for rank, (name, logit_val) in enumerate(
        sorted(contributions_to_token.items(), key=lambda x: -abs(x[1])), start=1
    ):
        pct = 100.0 * abs(logit_val) / total_abs
        dla_contributions[name] = {"logits": round(logit_val, 4), "percentage": round(pct, 2), "rank": rank}

    # Reorder by rank for top_10
    top_10 = sorted(dla_contributions.items(), key=lambda x: x[1]["rank"])[:10]
    top_10_list = [
        {"component": name, "logits": d["logits"], "percentage": d["percentage"]}
        for name, d in top_10
    ]

    total_attn = sum(contributions_to_token.get(f"L{i}_attn", 0) for i in range(n_layers))
    total_mlp = sum(contributions_to_token.get(f"L{i}_mlp", 0) for i in range(n_layers))
    embed_contribution = contributions_to_token.get("embed", 0.0)

    sum_to_chosen = sum(contributions_to_token.values())
    actual_chosen_logit = actual_logits[target_token_id].item()

    return {
        "chosen_token": chosen_token,
        "chosen_token_id": target_token_id,
        "dla_contributions": dla_contributions,
        "dla_summary": {
            "top_10_contributors": top_10_list,
            "total_attn_contribution": round(total_attn, 4),
            "total_mlp_contribution": round(total_mlp, 4),
            "embed_contribution": round(embed_contribution, 4),
        },
        "verification": {
            "passed": passed,
            "sum_of_contributions": round(sum_to_chosen, 6),
            "actual_logit": round(actual_chosen_logit, 6),
            "error": round(error, 6),
        },
    }


def verify_dla_decomposition(
    contributions: Dict[str, float],
    actual_logits: torch.Tensor,
    token_id: int,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Verify that the sum of DLA contributions to a token equals the actual logit.
    contributions: dict component_name -> logit contribution to token_id (scalar).
    actual_logits: full logit vector [vocab_size].
    """
    sum_contrib = sum(contributions.values())
    actual_val = actual_logits[token_id].item() if actual_logits.dim() >= 1 else actual_logits.item()
    error = abs(sum_contrib - actual_val)
    passed = error <= atol
    return {
        "passed": passed,
        "sum_of_contributions": sum_contrib,
        "actual_logit": actual_val,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Token Attribution: which INPUT tokens contributed to the output (DLA × Attention)
# ---------------------------------------------------------------------------

def _layer_from_component_name(component_name: str) -> Optional[int]:
    """Extract layer index from component name like 'L27_attn' -> 27."""
    if "_attn" not in component_name or not component_name.startswith("L"):
        return None
    try:
        return int(component_name[1 : component_name.index("_")])
    except (ValueError, TypeError):
        return None


def compute_token_attribution(
    dla_contributions: Dict[str, Dict[str, Any]],
    attention_from_last_per_layer: List[torch.Tensor],
    input_ids: torch.Tensor,
    tokenizer: Any,
    prompt_len: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute which input tokens contributed to the output token (DLA × attention).
    Only considers tokens from the original prompt (excludes thinking tokens after prompt_len).

    For each attention component, distribute its DLA contribution across heads and
    multiply by attention weights: contribution[position] += attn[head, pos] * (dla_attn / num_heads).

    Args:
        dla_contributions: DLA result["dla_contributions"] (e.g. {"L27_attn": {"logits": 2.3, ...}})
        attention_from_last_per_layer: List of [num_heads, seq_len] per layer (attention from last pos)
        input_ids: [batch, seq_len] token ids for the DLA input sequence
        tokenizer: For decoding token ids to strings
        prompt_len: Only consider positions [0 : prompt_len] (original prompt). If None, uses all positions.

    Returns:
        {"token_str": {"contribution": float, "percentage": float, "rank": int}, ...}
    """
    seq_len = input_ids.shape[1]
    if prompt_len is None:
        prompt_len = seq_len
    else:
        prompt_len = min(prompt_len, seq_len)
    ids_list = input_ids[0].tolist()
    # Token string per position (only for prompt positions)
    token_str_per_pos = []
    for pos in range(prompt_len):
        t = tokenizer.decode([ids_list[pos]], skip_special_tokens=False)
        token_str_per_pos.append(t.strip() or t)

    # Sum contribution per position (only for prompt positions)
    position_contributions = [0.0] * prompt_len
    for component_name, component_data in dla_contributions.items():
        layer = _layer_from_component_name(component_name)
        if layer is None or layer >= len(attention_from_last_per_layer):
            continue
        dla_logits = component_data.get("logits", 0.0)
        attn = attention_from_last_per_layer[layer]
        if attn.dim() == 2:
            num_heads, attn_seq_len = attn.shape
            head_dla = dla_logits / max(num_heads, 1)
            for head_idx in range(num_heads):
                for pos in range(min(attn_seq_len, prompt_len)):
                    position_contributions[pos] += attn[head_idx, pos].item() * head_dla
        else:
            continue

    # Aggregate by token string (sum contributions for same word at different positions)
    token_to_contribution: Dict[str, float] = {}
    for pos, token_str in enumerate(token_str_per_pos):
        key = token_str if token_str else f"<pos{pos}>"
        token_to_contribution[key] = token_to_contribution.get(key, 0.0) + position_contributions[pos]

    total_abs = sum(abs(v) for v in token_to_contribution.values())
    if total_abs < 1e-9:
        total_abs = 1.0
    result = {}
    for rank, (token_str, contrib) in enumerate(
        sorted(token_to_contribution.items(), key=lambda x: -abs(x[1])), start=1
    ):
        pct = 100.0 * abs(contrib) / total_abs
        result[token_str] = {
            "contribution": round(contrib, 4),
            "percentage": round(pct, 2),
            "rank": rank,
        }
    return result


# ---------------------------------------------------------------------------
# Find first token after thinking (</think> or end_of_thought) in generated token list
# ---------------------------------------------------------------------------

def find_think_end_token_index(
    generated_ids: List[int],
    tokenizer: Any,
) -> Optional[int]:
    """
    Find the index of the last token that is part of the thinking end marker
    (</think> or <|end_of_thought|>). First actual response token is at returned_index + 1.
    Returns None if no thinking block is found.
    """
    if not generated_ids:
        return None
    # 1. Try literal "</think>" tokenization (may be one or several tokens)
    end_marker_str = "</think>"
    try:
        end_ids = tokenizer.encode(end_marker_str, add_special_tokens=False)
    except Exception:
        end_ids = []
    if end_ids:
        n = len(end_ids)
        for i in range(len(generated_ids) - n + 1):
            if generated_ids[i : i + n] == end_ids:
                return i + n - 1  # index of last token of marker; first response = i + n
    # 2. Try special token e.g. <|end_of_thought|> (Qwen/DeepSeek style)
    decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    for tok_id, tok_val in decoder.items():
        tok_str = getattr(tok_val, "content", tok_val) if hasattr(tok_val, "content") else str(tok_val)
        if isinstance(tok_str, str) and ("end" in tok_str.lower() and "thought" in tok_str.lower()):
            if tok_id in generated_ids:
                return generated_ids.index(tok_id)
    # 3. Fallback: search by decoding and finding "</think>" in text
    for i in range(len(generated_ids)):
        seg = tokenizer.decode(generated_ids[: i + 1], skip_special_tokens=False)
        if end_marker_str in seg and seg.rstrip().endswith(end_marker_str):
            return i
    return None


# ---------------------------------------------------------------------------
# High-level API: from prompt string (for testing / trace integration)
# ---------------------------------------------------------------------------

def _prompt_to_inputs_with_chat_template(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    enable_thinking: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build input_ids using chat template (DeepSeek-R1 / Qwen reasoning format).
    Tries enable_thinking=True for reasoning mode; falls back if not supported.
    """
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None) is not None:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        text = prompt + "\n\nAssistant:"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        max_length=2048,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def compute_dla_for_prompt(
    model: torch.nn.Module,
    prompt: str,
    tokenizer: Any,
    target_token_id: Optional[int] = None,
    device: Optional[torch.device] = None,
    use_chat_template: bool = True,
) -> Dict[str, Any]:
    """
    Tokenize prompt, run forward, compute DLA for the predicted (or specified) next token.
    If use_chat_template=True (default), uses chat template for DeepSeek-R1 reasoning format.
    """
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        if use_chat_template:
            input_ids, attention_mask = _prompt_to_inputs_with_chat_template(
                tokenizer, prompt, device, enable_thinking=True
            )
        else:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
                max_length=2048,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
    return compute_dla_for_token(
        model,
        input_ids,
        attention_mask,
        tokenizer,
        target_token_id=target_token_id,
    )


if __name__ == "__main__":
    # Test: full generation (thinking + response), then DLA for first token of *response* only
    import sys
    import os
    import json
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from trace import load_model_and_tokenizer, _parse_thinking_from_output

    print("Loading model and tokenizer...")
    model, tokenizer, quant_info = load_model_and_tokenizer(use_4bit=True)
    print(f"Loaded: {quant_info}")

    prompt = "The Eiffel Tower is located in the city of _____. Answer in one word only."
    print(f"\nPrompt (with chat template / reasoning): {repr(prompt)}")

    device = next(model.parameters()).device
    with torch.no_grad():
        input_ids, attention_mask = _prompt_to_inputs_with_chat_template(
            tokenizer, prompt, device, enable_thinking=True
        )
        prompt_len = input_ids.shape[1]

        # 1. Generate full output (no token limit) so we get full thinking + full response
        print("\n--- Generating full output ---")
        gen = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )
        generated_part = gen[0, prompt_len:]
        generated_part_ids = generated_part.tolist()
        decoded_full = tokenizer.decode(generated_part, skip_special_tokens=False)
        thinking, response = _parse_thinking_from_output(decoded_full)

        # 2. Detect reasoning mode: find first token AFTER "</think>" (or <|end_of_thought|>)
        think_end_idx = find_think_end_token_index(generated_part_ids, tokenizer)
        if think_end_idx is not None:
            first_response_token_index = think_end_idx + 1
        else:
            first_response_token_index = 0  # No thinking mode

        # Skip whitespace/newline tokens so DLA analyzes first actual content token (e.g. " Paris")
        while first_response_token_index < len(generated_part_ids):
            tok_str = tokenizer.decode([generated_part_ids[first_response_token_index]], skip_special_tokens=False)
            if tok_str.strip():
                break
            first_response_token_index += 1
        if first_response_token_index >= len(generated_part_ids):
            first_response_token_index = 0

        thinking_token_ids = generated_part_ids[:first_response_token_index]
        response_token_ids = generated_part_ids[first_response_token_index:]

        # 3. Show thinking tokens (for debugging) and response separately
        print("\n=== Thinking tokens ===")
        if thinking_token_ids:
            thinking_decoded = tokenizer.decode(thinking_token_ids, skip_special_tokens=False)
            print(thinking_decoded if thinking_decoded.strip() else "(empty)")
        else:
            print("(none)")

        print("\n=== Response (actual output) ===")
        if response_token_ids:
            print(tokenizer.decode(response_token_ids, skip_special_tokens=False))
        else:
            print(response if response else decoded_full)

        # 4. First thinking token (debug) vs first actual response token
        first_thinking_token_id = int(generated_part_ids[0]) if generated_part_ids else None
        first_response_token_id = int(generated_part_ids[first_response_token_index])

        print("\n=== First thinking token (debug) ===")
        print(f"Token: {tokenizer.decode([first_thinking_token_id])!r} (id={first_thinking_token_id}) — inside <think>")

        print("\n=== First token of response (actual output) only ===")
        print(f"Token: {tokenizer.decode([first_response_token_id])!r} (after </think>)")
        print("DLA analyzes this token only (not thinking tokens).")

        # 5. Run DLA for the first token of the response only (first token after </think>)
        dla_input_len = prompt_len + first_response_token_index
        dla_input_ids = gen[0:1, :dla_input_len]
        dla_attention = torch.ones_like(dla_input_ids, device=device, dtype=torch.long)
        result = compute_dla_for_token(
            model,
            dla_input_ids,
            dla_attention,
            tokenizer,
            target_token_id=first_response_token_id,
        )
        # Token attribution (DLA × attention): which input words contributed
        if "error" not in result and "dla_contributions" in result:
            try:
                with torch.no_grad():
                    out_attn = model(
                        input_ids=dla_input_ids,
                        attention_mask=dla_attention,
                        output_hidden_states=False,
                        output_attentions=True,
                    )
                if getattr(out_attn, "attentions", None) and len(out_attn.attentions) > 0:
                    attention_from_last = [
                        out_attn.attentions[l][0, :, -1, :].cpu().float()
                        for l in range(len(out_attn.attentions))
                    ]
                    result["input_token_attribution"] = compute_token_attribution(
                        result["dla_contributions"],
                        attention_from_last,
                        dla_input_ids,
                        tokenizer,
                        prompt_len=prompt_len,  # Only consider original prompt tokens (exclude thinking)
                    )
            except Exception:
                result["input_token_attribution"] = {"error": "Could not compute"}

    if "error" in result and "final_residual" in str(result.get("error", "")):
        print("DLA capture failed:", result.get("error"))
    else:
        print("\n--- DLA: first token of response (actual output) only ---")
        print(f"Token: {result['chosen_token']!r} (id={result['chosen_token_id']})")
        print("\nTop 10 contributors:")
        for x in result["dla_summary"]["top_10_contributors"]:
            print(f"  {x['component']}: {x['logits']:+.2f} ({x['percentage']:.1f}%)")
        print("\nVerification:", result["verification"])
        if result["verification"]["passed"]:
            print("PASSED: sum of contributions matches actual logit within tolerance")
        else:
            print("FAILED: decomposition error > atol")
        ita = result.get("input_token_attribution")
        if ita and isinstance(ita, dict) and "error" not in ita:
            print("\n--- Input token attribution (which input words contributed) ---")
            top = sorted(ita.items(), key=lambda x: x[1].get("rank", 99))[:15]
            for token_str, data in top:
                print(f"  {data.get('rank', 0)}. {token_str!r}: {data.get('contribution', 0):+.2f} ({data.get('percentage', 0):.1f}%)")
        elif ita and isinstance(ita, dict) and ita.get("error"):
            print("\nToken attribution:", ita.get("error"))
        with open(os.path.join(os.path.dirname(__file__), "dla_test_output.json"), "w") as f:
            json.dump(result, f, indent=2)
        print("\nFull result written to dla_test_output.json")
