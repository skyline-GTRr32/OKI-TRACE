<div align="center">

# OKI TRACE: Step-by-step LLM Observability

**See what your LLM does at every step and every layer.**

![OKI TRACE](images/oki_trace_hero_image.png)

[![Repo views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/skyline-GTRr32/OKI-TRACE&icon=github.svg&icon_color=%23121011&title=views&count_bg=%2379C83D&title_bg=%23555555&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub stars](https://img.shields.io/github/stars/skyline-GTRr32/OKI-TRACE?style=flat-square)](https://github.com/skyline-GTRr32/OKI-TRACE/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/skyline-GTRr32/OKI-TRACE?style=flat-square)](https://github.com/skyline-GTRr32/OKI-TRACE/network/members)
[![GitHub license](https://img.shields.io/github/license/skyline-GTRr32/OKI-TRACE?style=flat-square)](https://github.com/skyline-GTRr32/OKI-TRACE/blob/main/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue?style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)

</div>

---

## The Thinking

When we send a prompt to an LLM, we get an answer—but we usually have no idea *how* it got there. If the answer is wrong or half-right, we can't tell whether the model misunderstood the prompt, got derailed mid-way, or was right until the last few tokens. We're flying blind.

**The core idea:** before we can interpret, steer, or improve how a model follows instructions, we need to **see what it actually does**—at every generation step and at every layer. Not as a black box, but as a trace: what it predicted, what it attended to, and how that prediction formed through the stack.

This project is about building that **observability layer** for local LLMs: capture and visualize what happens inside the model when it produces each token.

---

## What We're Building (v1)

**v1 has one goal:** given a model and a prompt, capture and display **what the LLM does at every step and every layer**.

### Every Step

A **step** is each time the model produces one new token. For every step we capture:

- **Chosen token** — The token the model generated.
- **Logits (top-k)** — The next-token distribution: what else it considered and with what probability.
- **Attention (Evidence)** — Which input tokens (from the prompt and previous output) received the most attention when producing this token. This is the model’s “evidence” for its choice.

### Every Layer

For that same forward pass, at **each transformer layer** we compute a **Logit Lens**: we take the hidden state at the position we’re predicting from, run it through the model’s output head, and get a distribution over the vocabulary. So for each layer we see:

> *“At this layer, the model would have predicted: token A (p₁), token B (p₂), …”*

That shows how the final answer **emerges through the layers**—from noisy early layers to the refined prediction at the end.

### The Dashboard

A **Streamlit dashboard** lets you:

- **Chat** with the model: send messages and see assistant replies. Each reply is traced.
- For each generated token: click to view that step’s **logits** (top-k), **attention** (Evidence), and **Logit Lens** (prediction at each layer).
- Inspect, layer by layer, how the model arrived at each token.

Everything runs **locally**. No telemetry, no cloud. Your model, your data, your machine.

---

## What v1 Is *Not*

- **Not** a user-facing SDK or integration API—we focus on the core “capture and display” first. Integration (e.g. “add one line to your code”) comes later.
- **Not** reports, analytics, or “improve your conversations”—that builds on top of this.
- **Not** gradient attribution, activation patching, or other advanced interpretability—v1 is logits, attention, and Logit Lens only.

---

## Tech (v1)

- **Models:** Any HuggingFace `AutoModelForCausalLM` (Qwen, Llama, Mistral, Phi, GPT-2, etc.). Set the Model ID in the dashboard or pass `model_id` to `load_model_and_tokenizer()`. 4-bit is optional with fallback to fp16/bf16.
- **Dashboard:** Streamlit, running locally. Model ID and 4-bit are in the sidebar.
- **Capture:** `output_attentions`, `output_hidden_states` (with fallbacks when a model or its attention backend does not support them). Logit Lens uses a detected final norm + `lm_head`; if not found, it is skipped. Chat: `apply_chat_template` when the tokenizer has a chat template, otherwise a simple `USER: ... ASSISTANT:` fallback.

---

## Summary

We’re building a **local, step-by-step and layer-by-layer trace** of what a language model does when it generates—and a **Streamlit dashboard** to see it. v1 is only that. When that works, we’ll add how others can plug it in and what comes next.
