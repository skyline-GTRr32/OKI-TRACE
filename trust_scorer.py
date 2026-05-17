"""
trust_scorer.py — Three-signal trust verdict for OKI-TRACE.

Combines entropy, MLP/attention gap, and response-level MLP ratio
into a single structured verdict with per-signal evidence.

No model access. No forward passes. Pure logic on existing trace data.

Thresholds (empirically derived from 9 labeled examples, DeepSeek-R1-Distill-Qwen-1.5B):
    EAS > 200 OR peak entropy > 6.0     → unstable computation
    MLP/attn gap > 20%                  → retrieved
    MLP/attn gap 10-20%                 → borderline
    MLP/attn gap < 10%                  → reconstructed
    Response MLP ratio < 65%            → attention drift, downgrade one level

NOTE: These thresholds are calibrated on one 1.5B model and 9 data points.
      They are a starting point, not a final answer. Recalibrate as data grows.
"""

from __future__ import annotations
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Thresholds — change these as more data is collected
# ---------------------------------------------------------------------------

ENTROPY_EAS_THRESHOLD       = 200.0   # Mean EAS above this → unstable
ENTROPY_PEAK_THRESHOLD      = 6.0     # Peak token entropy above this → unstable
GAP_RETRIEVED_THRESHOLD     = 20.0    # MLP/attn gap above this → retrieved
GAP_BORDERLINE_THRESHOLD    = 10.0    # MLP/attn gap above this → borderline (else reconstructed)
MLP_RATIO_THRESHOLD         = 65.0    # Response MLP ratio below this → attention drift


# ---------------------------------------------------------------------------
# Signal 1 — Entropy gate
# ---------------------------------------------------------------------------

def _score_entropy(entropy_summary: Dict) -> Dict[str, Any]:
    """
    Signal 1: Is the model's computation stable?

    Uses Mean EAS and Peak Token Entropy from entropy_summary.
    If either exceeds threshold → computation was unstable.

    Returns:
        result: "pass" | "fail"
        reason: human-readable explanation
        mean_eas: float
        peak_entropy: float
    """
    if not entropy_summary:
        return {
            "result": "unavailable",
            "reason": "Entropy data not present in trace (run with entropy enabled)",
            "mean_eas": None,
            "peak_entropy": None,
        }

    mean_eas = entropy_summary.get("mean_eas")
    peak_entropy = entropy_summary.get("max_token_entropy")

    if mean_eas is None and peak_entropy is None:
        return {
            "result": "unavailable",
            "reason": "EAS and peak entropy both missing",
            "mean_eas": None,
            "peak_entropy": None,
        }

    eas_failed = mean_eas is not None and mean_eas > ENTROPY_EAS_THRESHOLD
    peak_failed = peak_entropy is not None and peak_entropy > ENTROPY_PEAK_THRESHOLD

    if eas_failed and peak_failed:
        reason = (
            f"Mean EAS {mean_eas:.1f} > {ENTROPY_EAS_THRESHOLD} "
            f"AND peak entropy {peak_entropy:.2f} > {ENTROPY_PEAK_THRESHOLD} — "
            "computation was deeply unstable throughout"
        )
    elif eas_failed:
        reason = (
            f"Mean EAS {mean_eas:.1f} > {ENTROPY_EAS_THRESHOLD} — "
            "uncertainty persisted across layers, never resolved cleanly"
        )
    elif peak_failed:
        reason = (
            f"Peak entropy {peak_entropy:.2f} > {ENTROPY_PEAK_THRESHOLD} — "
            "at least one token had extremely high uncertainty"
        )
    else:
        reason = (
            f"Mean EAS {mean_eas:.1f} <= {ENTROPY_EAS_THRESHOLD}, "
            f"peak entropy {peak_entropy:.2f} <= {ENTROPY_PEAK_THRESHOLD} — "
            "computation appears stable"
        )

    return {
        "result": "fail" if (eas_failed or peak_failed) else "pass",
        "reason": reason,
        "mean_eas": round(mean_eas, 2) if mean_eas is not None else None,
        "peak_entropy": round(peak_entropy, 4) if peak_entropy is not None else None,
    }


# ---------------------------------------------------------------------------
# Signal 2 — MLP / Attention gap
# ---------------------------------------------------------------------------

def _score_mlp_attn_gap(dla_first_response: Dict) -> Dict[str, Any]:
    """
    Signal 2: Did the first response token come from retrieval or reconstruction?

    Computes: top_mlp_component% - top_attn_component%
    from dla_first_response["dla_summary"]["top_10_contributors"].

    High gap → one MLP dominated, attention was minor → retrieval.
    Low gap  → attention climbed near MLP level → reconstruction.

    Returns:
        result: "pass" | "warn" | "fail" | "unavailable"
        verdict: "retrieved" | "borderline" | "reconstructed" | "unavailable"
        gap: float
        top_mlp_component: str
        top_mlp_pct: float
        top_attn_component: str
        top_attn_pct: float
        reason: str
    """
    if not dla_first_response or "error" in dla_first_response:
        return {
            "result": "unavailable",
            "verdict": "unavailable",
            "gap": None,
            "top_mlp_component": None,
            "top_mlp_pct": None,
            "top_attn_component": None,
            "top_attn_pct": None,
            "reason": "DLA first response token not available",
        }

    summary = dla_first_response.get("dla_summary", {})
    contributors = summary.get("top_10_contributors", [])

    if not contributors:
        return {
            "result": "unavailable",
            "verdict": "unavailable",
            "gap": None,
            "top_mlp_component": None,
            "top_mlp_pct": None,
            "top_attn_component": None,
            "top_attn_pct": None,
            "reason": "No DLA contributors found",
        }

    # Find top MLP and top attention component by percentage
    top_mlp = None
    top_attn = None
    for c in contributors:
        name = c.get("component", "")
        pct = c.get("percentage", 0.0)
        if "_mlp" in name:
            if top_mlp is None or pct > top_mlp["percentage"]:
                top_mlp = {"component": name, "percentage": pct}
        elif "_attn" in name:
            if top_attn is None or pct > top_attn["percentage"]:
                top_attn = {"component": name, "percentage": pct}

    if top_mlp is None:
        return {
            "result": "unavailable",
            "verdict": "unavailable",
            "gap": None,
            "top_mlp_component": None,
            "top_mlp_pct": None,
            "top_attn_component": None,
            "top_attn_pct": top_attn["percentage"] if top_attn else None,
            "reason": "No MLP component found in top 10 contributors",
        }

    top_attn_pct = top_attn["percentage"] if top_attn else 0.0
    top_attn_name = top_attn["component"] if top_attn else "none"
    gap = top_mlp["percentage"] - top_attn_pct

    if gap > GAP_RETRIEVED_THRESHOLD:
        result = "pass"
        verdict = "retrieved"
        reason = (
            f"{top_mlp['component']} dominated at {top_mlp['percentage']:.1f}%, "
            f"top attention ({top_attn_name}) only {top_attn_pct:.1f}% — "
            f"gap {gap:.1f}% > {GAP_RETRIEVED_THRESHOLD}% — strong MLP retrieval"
        )
    elif gap > GAP_BORDERLINE_THRESHOLD:
        result = "warn"
        verdict = "borderline"
        reason = (
            f"{top_mlp['component']} at {top_mlp['percentage']:.1f}%, "
            f"top attention ({top_attn_name}) at {top_attn_pct:.1f}% — "
            f"gap {gap:.1f}% in borderline zone ({GAP_BORDERLINE_THRESHOLD}-{GAP_RETRIEVED_THRESHOLD}%)"
        )
    else:
        result = "fail"
        verdict = "reconstructed"
        reason = (
            f"{top_mlp['component']} at {top_mlp['percentage']:.1f}%, "
            f"top attention ({top_attn_name}) at {top_attn_pct:.1f}% — "
            f"gap {gap:.1f}% < {GAP_BORDERLINE_THRESHOLD}% — attention infiltrated, reconstruction likely"
        )

    return {
        "result": result,
        "verdict": verdict,
        "gap": round(gap, 2),
        "top_mlp_component": top_mlp["component"],
        "top_mlp_pct": round(top_mlp["percentage"], 2),
        "top_attn_component": top_attn_name,
        "top_attn_pct": round(top_attn_pct, 2),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Signal 3 — Response-level MLP ratio
# ---------------------------------------------------------------------------

def _score_mlp_ratio(dla_response_summary: Dict) -> Dict[str, Any]:
    """
    Signal 3: Was MLP dominant across the full response?

    Uses dla_response_summary["mlp_ratio"].
    Below threshold → attention was unusually involved throughout response.

    Returns:
        result: "pass" | "fail" | "unavailable"
        mlp_ratio: float
        reason: str
    """
    if not dla_response_summary:
        return {
            "result": "unavailable",
            "mlp_ratio": None,
            "reason": "DLA response summary not available",
        }

    mlp_ratio = dla_response_summary.get("mlp_ratio")
    if mlp_ratio is None:
        return {
            "result": "unavailable",
            "mlp_ratio": None,
            "reason": "MLP ratio not computed in response summary",
        }

    if mlp_ratio < MLP_RATIO_THRESHOLD:
        result = "fail"
        reason = (
            f"Response MLP ratio {mlp_ratio:.1f}% < {MLP_RATIO_THRESHOLD}% — "
            "attention was unusually involved throughout the full response"
        )
    else:
        result = "pass"
        reason = (
            f"Response MLP ratio {mlp_ratio:.1f}% >= {MLP_RATIO_THRESHOLD}% — "
            "MLP dominated as expected for factual generation"
        )

    return {
        "result": result,
        "mlp_ratio": round(mlp_ratio, 1),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Verdict logic — combine three signals
# ---------------------------------------------------------------------------

def _compute_verdict(
    entropy_signal: Dict,
    gap_signal: Dict,
    ratio_signal: Dict,
) -> tuple[str, str, int]:
    """
    Combine three signals into final verdict.

    Rules:
    1. If entropy FAILS → verdict is "Unstable" regardless of other signals
    2. Count fails + warns from gap and ratio signals
    3. Apply downgrade from ratio if it fails
    4. Map to verdict label

    Returns: (verdict, recommendation, flags_fired)
    """
    # Rule 1 — entropy gate
    if entropy_signal.get("result") == "fail":
        return (
            "Unstable",
            "Model computation was unstable. Output unreliable — do not trust.",
            3,
        )

    # Count signal 2 and 3 flags
    gap_result = gap_signal.get("result", "unavailable")
    gap_verdict = gap_signal.get("verdict", "unavailable")
    ratio_result = ratio_signal.get("result", "unavailable")

    flags = 0
    if gap_result == "fail":
        flags += 2      # hard fail counts double
    elif gap_result == "warn":
        flags += 1

    if ratio_result == "fail":
        flags += 1      # ratio is a downgrade signal

    # Map flags to verdict
    if flags == 0:
        verdict = "Retrieved"
        recommendation = "Internal signals consistent with clean memory retrieval. High trust."
    elif flags == 1:
        verdict = "Borderline"
        recommendation = "One signal flagged. Probably correct but verify if this output is critical."
    elif flags == 2:
        verdict = "Borderline"
        recommendation = "Two signals flagged. Treat with caution and verify independently."
    else:
        verdict = "Reconstructed"
        recommendation = (
            "Multiple signals flagged. Model likely reconstructed this output "
            "from weak signals rather than retrieving a stored fact. "
            "Verify this output before using it."
        )

    return verdict, recommendation, flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_trust_verdict(trace: Dict) -> Dict[str, Any]:
    """
    Compute three-signal trust verdict from a completed trace dict.

    Args:
        trace: the full trace dict returned by run_traced().
               Must contain: entropy_summary, dla_first_response,
               dla_response_summary (all present when respective
               features are enabled).

    Returns:
        dict with:
            verdict        : "Retrieved" | "Borderline" | "Reconstructed" | "Unstable"
            recommendation : str — human-readable action
            flags_fired    : int — how many signals flagged (0-3)
            signals        : {
                entropy    : signal 1 result dict
                mlp_attn_gap : signal 2 result dict
                mlp_ratio  : signal 3 result dict
            }
            thresholds_used : dict — what thresholds were applied
            data_warning    : str | None — warning if insufficient signals available
    """
    entropy_summary      = trace.get("entropy_summary", {})
    dla_first_response   = trace.get("dla_first_response", {})
    dla_response_summary = trace.get("dla_response_summary", {})

    entropy_signal = _score_entropy(entropy_summary)
    gap_signal     = _score_mlp_attn_gap(dla_first_response)
    ratio_signal   = _score_mlp_ratio(dla_response_summary)

    verdict, recommendation, flags_fired = _compute_verdict(
        entropy_signal, gap_signal, ratio_signal
    )

    # Data warning if signals unavailable
    unavailable = [
        name for name, sig in [
            ("entropy", entropy_signal),
            ("mlp_attn_gap", gap_signal),
            ("mlp_ratio", ratio_signal),
        ]
        if sig.get("result") == "unavailable"
    ]
    data_warning = (
        f"Signals unavailable: {', '.join(unavailable)}. "
        "Run with compute_dla=True and entropy enabled for full verdict."
        if unavailable else None
    )

    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "flags_fired": flags_fired,
        "signals": {
            "entropy":     entropy_signal,
            "mlp_attn_gap": gap_signal,
            "mlp_ratio":   ratio_signal,
        },
        "thresholds_used": {
            "eas_threshold":          ENTROPY_EAS_THRESHOLD,
            "peak_entropy_threshold": ENTROPY_PEAK_THRESHOLD,
            "gap_retrieved":          GAP_RETRIEVED_THRESHOLD,
            "gap_borderline":         GAP_BORDERLINE_THRESHOLD,
            "mlp_ratio_threshold":    MLP_RATIO_THRESHOLD,
        },
        "data_warning": data_warning,
    }
