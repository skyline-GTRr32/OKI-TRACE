"""
dla_response.py — Response-level DLA aggregation for OKI-TRACE.

Takes the per-token DLA already stored in every step_dict and aggregates
it across the full response into a single layer contribution summary.

This answers: "Across the entire output, which layers contributed how much?"

No model access. No forward passes. Pure math on existing trace data.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional


def compute_response_dla_summary(steps: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate per-token DLA contributions across all response steps.

    For each component (e.g. L27_mlp, L26_attn), sum the absolute logit
    contributions across every token in the response, then normalize to
    percentages.

    Also computes:
    - MLP vs attention ratio across full response
    - Top-1 concentration trajectory (per step) for drift detection
    - Per-layer total contribution (summed across attn + mlp per layer)

    Args:
        steps: list of step_dicts from trace["steps"].
               Each step must have a "dla" key containing the output of
               compute_dla_for_token() — specifically "dla_contributions".

    Returns:
        dict with:
            "component_totals"     : {component: {"total_abs": float, "percentage": float}}
            "top_10_components"    : list of top 10 components by total abs contribution
            "layer_totals"         : {layer_idx: {"mlp": float, "attn": float, "total": float}}
            "mlp_ratio"            : float — % of total from MLP components
            "attn_ratio"           : float — % of total from attention components
            "concentration_trajectory": list of {"step": int, "token": str, "top1_component": str, "top1_pct": float}
            "tokens_analyzed"      : int — how many steps had valid DLA
            "tokens_total"         : int — total steps
    """
    component_totals: Dict[str, float] = {}
    concentration_trajectory = []
    tokens_analyzed = 0
    tokens_total = len(steps)

    for s in steps:
        dla = s.get("dla")
        if dla is None or "error" in dla or "dla_contributions" not in dla:
            continue

        contributions = dla["dla_contributions"]  # {component: {"logits": float, "percentage": float, "rank": int}}
        tokens_analyzed += 1

        # Accumulate absolute logit contributions per component
        for component, data in contributions.items():
            abs_val = abs(data.get("logits", 0.0))
            component_totals[component] = component_totals.get(component, 0.0) + abs_val

        # Concentration trajectory — top-1 component for this step
        if contributions:
            top1 = min(contributions.items(), key=lambda x: x[1].get("rank", 99))
            concentration_trajectory.append({
                "step": s.get("step", 0),
                "token": s.get("chosen_token", ""),
                "top1_component": top1[0],
                "top1_pct": top1[1].get("percentage", 0.0),
            })

    if not component_totals:
        return {
            "component_totals": {},
            "top_10_components": [],
            "layer_totals": {},
            "mlp_ratio": None,
            "attn_ratio": None,
            "concentration_trajectory": [],
            "tokens_analyzed": 0,
            "tokens_total": tokens_total,
        }

    # Normalize component totals to percentages
    grand_total = sum(component_totals.values())
    if grand_total < 1e-9:
        grand_total = 1.0

    component_pct: Dict[str, Dict] = {}
    for comp, total in component_totals.items():
        component_pct[comp] = {
            "total_abs": round(total, 4),
            "percentage": round(100.0 * total / grand_total, 2),
        }

    # Top 10 by total absolute contribution
    top_10 = sorted(component_pct.items(), key=lambda x: -x[1]["total_abs"])[:10]
    top_10_list = [
        {"component": name, "total_abs": d["total_abs"], "percentage": d["percentage"]}
        for name, d in top_10
    ]

    # Per-layer totals — group attn + mlp per layer index
    layer_totals: Dict[int, Dict[str, float]] = {}
    total_mlp = 0.0
    total_attn = 0.0

    for comp, total in component_totals.items():
        if comp == "embed":
            continue
        # Parse layer index from "L27_mlp" or "L27_attn"
        try:
            parts = comp.split("_")
            layer_idx = int(parts[0][1:])   # "L27" -> 27
            comp_type = parts[1]            # "mlp" or "attn"
        except (IndexError, ValueError):
            continue

        if layer_idx not in layer_totals:
            layer_totals[layer_idx] = {"mlp": 0.0, "attn": 0.0, "total": 0.0}

        layer_totals[layer_idx][comp_type] = layer_totals[layer_idx].get(comp_type, 0.0) + total
        layer_totals[layer_idx]["total"] += total

        if comp_type == "mlp":
            total_mlp += total
        elif comp_type == "attn":
            total_attn += total

    # MLP vs attn ratio
    mlp_attn_total = total_mlp + total_attn
    mlp_ratio = round(100.0 * total_mlp / mlp_attn_total, 1) if mlp_attn_total > 0 else None
    attn_ratio = round(100.0 * total_attn / mlp_attn_total, 1) if mlp_attn_total > 0 else None

    # Normalize layer totals to percentages within layer
    layer_totals_normalized = {}
    for idx, vals in sorted(layer_totals.items()):
        lt = vals["total"]
        layer_totals_normalized[idx] = {
            "mlp": round(vals.get("mlp", 0.0), 4),
            "attn": round(vals.get("attn", 0.0), 4),
            "total": round(lt, 4),
            "percentage": round(100.0 * lt / grand_total, 2),
        }

    return {
        "component_totals": component_pct,
        "top_10_components": top_10_list,
        "layer_totals": layer_totals_normalized,
        "mlp_ratio": mlp_ratio,
        "attn_ratio": attn_ratio,
        "concentration_trajectory": concentration_trajectory,
        "tokens_analyzed": tokens_analyzed,
        "tokens_total": tokens_total,
    }
