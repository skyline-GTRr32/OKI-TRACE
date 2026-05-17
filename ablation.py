"""
Head and layer ablation for HuggingFace CausalLM (Qwen/DeepSeek/Llama/Mistral layout).

Ablation zeroes out the contribution of specific attention heads or entire
attention/MLP layers — without touching model weights. Done via forward hooks
registered before a run and removed cleanly after.

Usage:
    from ablation import AblationConfig, apply_ablation_hooks, get_model_dims

    info = get_model_dims(model)
    # -> {"num_layers": 28, "num_heads": 16, "head_dim": 128}

    cfg = AblationConfig(
        disabled_heads={5: {0, 3}, 12: {7}},
        disabled_attn_layers={20},
        disabled_mlp_layers={3},
    )

    handles = apply_ablation_hooks(model, cfg)
    try:
        output = model(...)
    finally:
        for h in handles:
            h.remove()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """
    Specifies what to ablate.

    disabled_heads      : {layer_idx: set of head indices to zero out}
    disabled_attn_layers: set of layer indices where the FULL attn output is zeroed
    disabled_mlp_layers : set of layer indices where the FULL mlp output is zeroed
    """
    disabled_heads: Dict[int, Set[int]] = field(default_factory=dict)
    disabled_attn_layers: Set[int] = field(default_factory=set)
    disabled_mlp_layers: Set[int] = field(default_factory=set)

    def is_empty(self) -> bool:
        return (
            not self.disabled_heads
            and not self.disabled_attn_layers
            and not self.disabled_mlp_layers
        )

    def summary(self) -> str:
        parts = []
        for layer, heads in sorted(self.disabled_heads.items()):
            parts.append(f"L{layer} heads {sorted(heads)}")
        for layer in sorted(self.disabled_attn_layers):
            parts.append(f"L{layer} attn")
        for layer in sorted(self.disabled_mlp_layers):
            parts.append(f"L{layer} mlp")
        return ", ".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def _get_layers(model: torch.nn.Module):
    """Return the list of transformer layers for Qwen/Llama/GPT-2 style models."""
    inner = getattr(model, "model", None) or getattr(model, "transformer", model)
    layers = getattr(inner, "layers", None) or getattr(inner, "h", None)
    if layers is None:
        raise ValueError(
            "Cannot find model layers. Expected model.model.layers (Qwen/Llama) "
            "or model.transformer.h (GPT-2)."
        )
    return layers


def get_model_dims(model: torch.nn.Module) -> Dict[str, int]:
    """
    Detect num_layers, num_heads, head_dim, hidden_size from the model config.
    Falls back to weight shape inspection if config attrs are missing.
    """
    cfg = getattr(model, "config", None)
    num_layers = num_heads = head_dim = hidden_size = None

    if cfg is not None:
        num_layers = (
            getattr(cfg, "num_hidden_layers", None)
            or getattr(cfg, "n_layer", None)
            or getattr(cfg, "num_layers", None)
        )
        num_heads = (
            getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "n_head", None)
            or getattr(cfg, "num_heads", None)
        )
        hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", None)
            or getattr(cfg, "n_embd", None)
        )
        head_dim = getattr(cfg, "head_dim", None)

    try:
        layers = _get_layers(model)
        if num_layers is None:
            num_layers = len(layers)
        attn0 = layers[0].self_attn
        q_proj = getattr(attn0, "q_proj", None)
        if q_proj is not None and hasattr(q_proj, "weight"):
            out_dim = q_proj.weight.shape[0]
            if num_heads and head_dim is None:
                head_dim = out_dim // num_heads
            elif head_dim and num_heads is None:
                num_heads = out_dim // head_dim
    except Exception:
        pass

    if hidden_size and num_heads and head_dim is None:
        head_dim = hidden_size // num_heads

    return {
        "num_layers": int(num_layers) if num_layers else 0,
        "num_heads": int(num_heads) if num_heads else 0,
        "head_dim": int(head_dim) if head_dim else 0,
        "hidden_size": int(hidden_size) if hidden_size else 0,
    }


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _full_zero_hook(module, input, output):
    """Post hook: zero the entire module output (for full attn/mlp layer ablation)."""
    out = output[0] if isinstance(output, tuple) else output
    result = torch.zeros_like(out)
    if isinstance(output, tuple):
        return (result,) + output[1:]
    return result


def _make_head_pre_hook(disabled_heads: Set[int], num_heads: int, head_dim: int):
    """
    Pre hook on o_proj: zero the [batch, seq, head*head_dim] slices for disabled heads
    in the concatenated QKV-output tensor, BEFORE the output projection runs.
    This correctly removes those heads' contribution from the residual stream.
    """
    def hook(module, input):
        if not input or input[0].dim() != 3:
            return input
        x = input[0].clone()
        for h in disabled_heads:
            if h < num_heads:
                x[:, :, h * head_dim : (h + 1) * head_dim] = 0.0
        return (x,) + input[1:]
    return hook


def _make_head_fallback_post_hook(disabled_heads: Set[int], num_heads: int, head_dim: int):
    """
    Fallback post hook on self_attn when o_proj is not found.
    Zeroes head-sized slices of the output hidden dim (approximate after mixing).
    """
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        out = out.clone()
        for h in disabled_heads:
            if h < num_heads:
                start, end = h * head_dim, (h + 1) * head_dim
                if end <= out.shape[-1]:
                    out[:, :, start:end] = 0.0
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out
    return hook


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def apply_ablation_hooks(
    model: torch.nn.Module,
    config: AblationConfig,
) -> List[Any]:
    """
    Register all ablation hooks defined in config.
    Returns a list of hook handles — caller MUST call h.remove() on each when done.
    """
    if config.is_empty():
        return []

    dims = get_model_dims(model)
    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]

    if num_heads == 0 or head_dim == 0:
        raise ValueError(
            f"Could not detect num_heads/head_dim. Got: {dims}. "
            "Check that the model config has num_attention_heads and hidden_size."
        )

    layers = _get_layers(model)
    handles: List[Any] = []

    # All layers that need any attention hook
    attn_layers_needed = set(config.disabled_heads.keys()) | config.disabled_attn_layers

    for layer_idx, layer in enumerate(layers):
        # --- Attention ---
        if layer_idx in attn_layers_needed:
            attn = layer.self_attn

            if layer_idx in config.disabled_attn_layers:
                # Zero the entire attention output (overrides any head config for this layer)
                h = attn.register_forward_hook(_full_zero_hook)
                handles.append(h)

            elif layer_idx in config.disabled_heads:
                heads = config.disabled_heads[layer_idx]
                o_proj = getattr(attn, "o_proj", None)

                if o_proj is not None:
                    # Clean path: zero head slices before output projection
                    h = o_proj.register_forward_pre_hook(
                        _make_head_pre_hook(heads, num_heads, head_dim)
                    )
                else:
                    # Fallback: post hook on self_attn
                    h = attn.register_forward_hook(
                        _make_head_fallback_post_hook(heads, num_heads, head_dim)
                    )
                handles.append(h)

        # --- MLP ---
        if layer_idx in config.disabled_mlp_layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                h = mlp.register_forward_hook(_full_zero_hook)
                handles.append(h)

    return handles


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class ablation_context:
    """
    Context manager: applies hooks on __enter__, removes on __exit__.

        with ablation_context(model, config):
            output = model(input_ids=...)
    """
    def __init__(self, model: torch.nn.Module, config: AblationConfig):
        self.model = model
        self.config = config
        self.handles: List[Any] = []

    def __enter__(self):
        self.handles = apply_ablation_hooks(self.model, self.config)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []
