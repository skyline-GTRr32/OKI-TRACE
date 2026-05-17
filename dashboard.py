"""
Streamlit dashboard for the v1 trace: chat with the model, view logits, Evidence,
and Logit Lens per step for any assistant reply.

Run from project root (venv activated):
    streamlit run dashboard.py
"""

import json
import streamlit as st
from trace import load_model_and_tokenizer, run_traced, DEFAULT_MODEL_ID

st.set_page_config(page_title="Trace v1", layout="wide")
st.title("Trace v1 — Chat & inspect what the LLM does at every step")

# --- Session state ---
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {role, content} or {role, content, "trace": trace} for assistant
if "trace" not in st.session_state:
    st.session_state.trace = None  # trace to view (from "View trace" or file upload)

# --- Sidebar ---
with st.sidebar:
    st.header("Model")
    model_id = st.text_input(
        "Model ID",
        value=DEFAULT_MODEL_ID,
        key="trace_model_id",
        help="Any HuggingFace CausalLM: Qwen, Llama, Mistral, GPT-2, Phi, etc.",
    )
    use_4bit = st.checkbox("Use 4-bit quantization", value=True, key="trace_use_4bit", help="Fallback to fp16/bf16 if 4-bit fails")
    if st.session_state.model is not None and st.session_state.get("loaded_quantization"):
        st.success(f"**Loaded:** {st.session_state.get('loaded_quantization', '—')}")
    st.caption("6GB VRAM: first load can take 1–2 min; wait until you see \"Running traced generation...\".")
    if st.button("Reload model"):
        st.session_state.model = None
        st.session_state.tokenizer = None
        st.session_state.loaded_quantization = None
        st.rerun()

    st.header("Settings")
    enable_thinking = st.checkbox(
        "Enable thinking",
        value=False,
        key="trace_enable_thinking",
        help="DeepSeek-R1 / Qwen3: native thinking (<think> blocks). Other models: prompt to output <think>...</think> for reasoning.",
    )
    st.caption("Default model (DeepSeek-R1-Distill-Qwen-7B) supports reasoning; thinking is parsed from <think> blocks.")
    compute_dla = st.checkbox(
        "Compute DLA (first response token)",
        value=False,
        key="trace_compute_dla",
        help="DLA for first token of actual response only (after </think>, skips thinking tokens).",
    )
    max_new_tokens = st.number_input("Max new tokens", min_value=8, max_value=5000, value=512)
    if st.button("Clear chat"):
        st.session_state.chat = []
        st.session_state.trace = None
        st.rerun()

    st.divider()
    st.subheader("Or load trace from file")
    uploaded = st.file_uploader("trace JSON", type=["json"], key="upload_trace")

# --- Load trace from file ---
if uploaded is not None:
    try:
        st.session_state.trace = json.load(uploaded)
        st.sidebar.success("Trace loaded.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")

# --- Helpers for trace view (with backward compat) ---
def evidence_label(ev, prompt_len):
    pos = ev.get("position", 0)
    plen = prompt_len
    if "in_prompt" in ev and "relative" in ev:
        if ev["in_prompt"]:
            return f"pos {pos} (prompt)"
        rel = ev["relative"]
        return f"pos {pos} (generated, +{rel})"
    in_prompt = pos < plen
    if in_prompt:
        return f"pos {pos} (prompt)"
    return f"pos {pos} (generated, +{pos - plen})"

def render_trace_view(trace):
    """Render the step selector, Logits, Evidence, Logit Lens, and provenance."""
    prompt_len = trace.get("prompt_len", 0)
    provenance = trace.get("evidence_provenance", "Attention from final layer (heads averaged)")

    st.info(
        "Special tokens (e.g. `<|im_start|>`, `[INST]`) depend on the model's chat format. "
        "High attention to them is normal."
    )

    st.subheader("Prompt (full input to model)")
    st.text(trace.get("prompt", ""))

    st.subheader("Thinking (reasoning)")
    if trace.get("thinking"):
        st.markdown(trace.get("thinking", ""))
    else:
        st.caption("(none)")

    st.subheader("Response (actual output)")
    st.text(trace.get("output", "") or "(empty)")
    st.divider()

    steps = trace.get("steps", [])
    if not steps:
        st.warning("No steps in trace.")
        return

    options = [f"Step {s['step']}: {repr(s['chosen_token'])}" for s in steps]
    idx = st.selectbox("Select a step (generated token)", range(len(steps)), format_func=lambda i: options[i])
    s = steps[idx]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Logits (top-k)")
        for x in s["logits_topk"]:
            st.text(f"  {x['token']!r}  {x['prob']:.4f}")

    with c2:
        st.markdown("#### Evidence")
        st.caption(provenance)
        for x in s["evidence"]:
            lbl = evidence_label(x, prompt_len)
            st.text(f"  {x['token']!r}  w={x['weight']:.4f}  {lbl}")

    with c3:
        st.markdown("#### Logit Lens (first 8 layers)")
        if s["logit_lens"]:
            for ll in s["logit_lens"][:8]:
                row = " | ".join(f"{t['token']!r}({t['prob']})" for t in ll["topk"])
                st.text(f"  L{ll['layer']}: {row}")
        else:
            st.caption("Not available for this model")

    # --- DLA from step (backward compat for old traces) ---
    if s.get("dla") and "error" not in s["dla"]:
        st.divider()
        st.subheader("DLA — Direct Logit Attribution (this step)")
        d = s["dla"]
        st.caption(f"Chosen token: {d.get('chosen_token', '')!r} (id={d.get('chosen_token_id', '')})")
        if "dla_summary" in d and "top_10_contributors" in d["dla_summary"]:
            st.markdown("**Top 10 contributors**")
            for x in d["dla_summary"]["top_10_contributors"]:
                st.text(f"  {x['component']}: {x['logits']:+.2f} ({x['percentage']:.1f}%)")
        if "verification" in d:
            v = d["verification"]
            st.success(f"Verification: passed={v.get('passed', False)}, error={v.get('error', '')}")

    # --- DLA: first token of response only (from dla_analyzer logic) ---
    d = trace.get("dla_first_response")
    if d is not None and "error" not in d:
        st.divider()
        st.subheader("DLA — First token of response (actual output) only")
        st.caption(f"Token: {d.get('chosen_token', '')!r} (id={d.get('chosen_token_id', '')}) — after </think>, excludes thinking")
        if "dla_summary" in d and "top_10_contributors" in d["dla_summary"]:
            st.markdown("**Top 10 components**")
            for x in d["dla_summary"]["top_10_contributors"]:
                st.text(f"  {x['component']}: {x['logits']:+.2f} ({x['percentage']:.1f}%)")
        if "verification" in d:
            v = d["verification"]
            st.success(f"Verification: passed={v.get('passed', False)}, error={v.get('error', '')}")

        # Token attribution: which input words contributed to this output token
        ita = d.get("input_token_attribution")
        if ita is not None and "error" not in ita and isinstance(ita, dict):
            st.markdown("**Input token attribution** (which input words contributed to output)")
            top_tokens = sorted(ita.items(), key=lambda x: x[1].get("rank", 99))[:15]
            for token_str, data in top_tokens:
                c = data.get("contribution", 0)
                p = data.get("percentage", 0)
                r = data.get("rank", 0)
                st.text(f"  {r}. {token_str!r}: {c:+.2f} ({p:.1f}%)")
        elif ita is not None and isinstance(ita, dict) and ita.get("error"):
            st.caption(f"Token attribution: {ita.get('error', '')}")
    elif d is not None and "error" in d:
        st.divider()
        st.subheader("DLA — First token of response")
        st.warning(f"DLA failed: {d.get('error', '')}")

    st.divider()
    st.markdown("#### Logit Lens — all layers")
    if s["logit_lens"]:
        n = max(len(ll["topk"]) for ll in s["logit_lens"])
        cols = ["Layer"] + [f"Rank {i+1}" for i in range(n)]
        rows = []
        for ll in s["logit_lens"]:
            toks = [f"{t['token']!r} ({t['prob']})" for t in ll["topk"]]
            while len(toks) < n:
                toks.append("—")
            rows.append([f"L{ll['layer']}"] + toks)
        head = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * (n + 1)) + "|"
        body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
        st.markdown(head + "\n" + sep + "\n" + body)
    else:
        if not trace.get("logit_lens_available", True):
            st.warning("Logit Lens not available for this model (norm/lm_head not detected).")
        else:
            st.caption("No layers in this trace.")

# --- Chat ---
st.subheader("Chat")
for i, msg in enumerate(st.session_state.chat):
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role):
        if role == "assistant" and msg.get("thinking"):
            with st.expander("Thinking", expanded=False):
                st.markdown(msg.get("thinking", ""))
        st.markdown(content)
        if role == "assistant" and msg.get("trace") is not None:
            if st.button("View trace", key=f"view_trace_{i}"):
                st.session_state.trace = msg["trace"]
                st.rerun()

# Chat input
user_input = st.chat_input("Message")
if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})
    chat_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]

    model_id = st.session_state.get("trace_model_id") or DEFAULT_MODEL_ID
    use_4bit = st.session_state.get("trace_use_4bit", True)
    need_load = st.session_state.model is None
    if not need_load and (st.session_state.get("loaded_model_id") != model_id or st.session_state.get("loaded_use_4bit") != use_4bit):
        st.session_state.model = None
        st.session_state.tokenizer = None
        need_load = True

    try:
        if need_load:
            with st.spinner("Loading model (first time can take 1–2 min on 6GB VRAM; wait for it to finish)..."):
                st.session_state.model, st.session_state.tokenizer, quant_info = load_model_and_tokenizer(model_id=model_id, use_4bit=use_4bit)
            st.session_state.loaded_model_id = model_id
            st.session_state.loaded_use_4bit = use_4bit
            st.session_state.loaded_quantization = quant_info
        with st.spinner("Running traced generation..."):
            out, tr = run_traced(
                messages=chat_messages,
                max_new_tokens=max_new_tokens,
                model=st.session_state.model,
                tokenizer=st.session_state.tokenizer,
                topk_logits=10,
                topk_evidence=10,
                topk_lens=5,
                enable_thinking=st.session_state.get("trace_enable_thinking", False),
                compute_dla=st.session_state.get("trace_compute_dla", False),
            )
        thinking = (tr or {}).get("thinking")
        st.session_state.chat.append({
            "role": "assistant",
            "content": out,
            "trace": tr,
            "thinking": thinking,
        })
        st.session_state.trace = tr
    except Exception as e:
        err_msg = str(e)
        st.session_state.chat.append({
            "role": "assistant",
            "content": f"[Error: {err_msg}]",
            "trace": None,
        })
        st.error(f"Model load or generation failed: {err_msg}")
    st.rerun()

# --- Trace view (from chat "View trace" or loaded file) ---
if st.session_state.trace is not None:
    st.divider()
    st.subheader("Trace")
    render_trace_view(st.session_state.trace)

    st.sidebar.download_button(
        "Download trace JSON",
        data=json.dumps(st.session_state.trace, indent=2, ensure_ascii=False),
        file_name="trace.json",
        mime="application/json",
        key="dl_trace",
    )
else:
    st.divider()
    st.info(
        "Send a message in the chat to generate a traced reply, then use **View trace** on an assistant message. "
        "Or load a trace JSON from the sidebar."
    )
