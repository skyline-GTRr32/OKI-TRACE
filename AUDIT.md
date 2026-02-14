# Code audit: README vs implementation & unused logic

## README vs implementation — verified

| README claim | Implementation | Status |
|-------------|----------------|--------|
| **Every step:** Chosen token, Logits (top-k), Attention (Evidence) | `trace.py`: `chosen_token`, `logits_topk`, `evidence` per step | OK |
| **Every layer:** Logit Lens (hidden state → norm → lm_head → vocab distribution) | `run_traced` builds `logit_lens` per layer from `out.hidden_states` + norm + lm_head | OK |
| **Dashboard:** Chat, view logits/Evidence/Logit Lens per step | `dashboard.py`: chat input, `render_trace_view` with step selector and 3 columns | OK |
| **Model ID** in dashboard or `load_model_and_tokenizer(model_id=...)` | Sidebar `trace_model_id`, `load_model_and_tokenizer(model_id=..., use_4bit=...)` | OK |
| **4-bit optional with fallback to fp16/bf16** | `trace.py`: try BitsAndBytesConfig, on exception load with `torch_dtype` | OK |
| **output_attentions / output_hidden_states with fallbacks** | Try eager attention; on forward failure retry with both False | OK |
| **Logit Lens:** detected norm + lm_head; skipped if not found | `_get_norm_and_lm_head`, `logit_lens_available`; empty `logit_lens` when unavailable | OK |
| **Chat:** apply_chat_template or USER/ASSISTANT fallback | `_format_messages_to_prompt` in `trace.py` | OK |

All README behavior is implemented.

---

## Unused / dead code

1. **`test_model.py` line 66**  
   - `full = tokenizer.decode(out[0], skip_special_tokens=True)`  
   - `full` is never used. Only `response` (decode of new tokens only) is printed.  
   - **Fix:** Remove the unused `full` assignment.

---

## Duplication (maintainability)

- **`DEFAULT_MODEL_ID`** (and cache dir pattern) appears in:
  - `trace.py` (source of truth for tracer/dashboard)
  - `download_model.py`
  - `test_model.py` (as `MODEL_ID`)
- If the default model is changed, it should be updated in all three. Consider importing from `trace` in `download_model.py` and `test_model.py` to avoid drift.

---

## Potential errors / edge cases

1. **`test_model.py` and chat template**  
   - Uses `tokenizer.apply_chat_template(...)` directly. Tokenizers without a chat template (e.g. some base models) can raise.  
   - The rest of the project uses `_format_messages_to_prompt` in `trace.py`, which has a fallback.  
   - **Risk:** Running `test_model.py` with a model that has no chat template can fail. Optional improvement: use `trace._format_messages_to_prompt` or duplicate the fallback in `test_model.py`.

2. **`test_trace.py` and empty steps**  
   - `s = trace["steps"][0]` assumes at least one step. If `steps` is empty (e.g. `max_new_tokens=0` or immediate EOS before any token), this raises `IndexError`.  
   - **Risk:** Low with current `max_new_tokens=20`. Optional: guard with `if not trace["steps"]: return` or similar before indexing.

3. **`download_model.py` vs `trace.py` default model**  
   - Both define their own default model id. If one is changed and the other is not, behavior can diverge. Prefer a single source (e.g. `trace.DEFAULT_MODEL_ID`) in `download_model.py`.

---

## Functions and variables usage

- **trace.py:** `_get_norm_and_lm_head`, `_format_messages_to_prompt`, `load_model_and_tokenizer`, `run_traced` — all used (by `run_traced`, dashboard, or test_trace).
- **dashboard.py:** `evidence_label`, `render_trace_view` — used in trace view.
- **download_model.py:** `main()` — used when run as script.
- **test_trace.py:** `main()` — used when run as script.
- **test_model.py:** `main()` — used when run as script; `full` — unused (see above).

No other unused public or helper functions were found.
