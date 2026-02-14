"""
Run the tracer and print a summary of the trace plus a sample step.
Saves a short trace to trace_sample.json for inspection.

Run from project root (venv activated):
    python test_trace.py
"""

import json
from trace import load_model_and_tokenizer, run_traced

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    prompt = "Write a one-line Python function that adds two numbers."
    max_new_tokens = 20
    print(f"Running traced generation: max_new_tokens={max_new_tokens}")
    output_text, trace = run_traced(
        prompt,
        max_new_tokens=max_new_tokens,
        model=model,
        tokenizer=tokenizer,
        topk_logits=5,
        topk_evidence=5,
        topk_lens=3,
    )

    print()
    print("=" * 50)
    print("TRACE SUMMARY")
    print("=" * 50)
    print("model_id:", trace["model_id"])
    print("prompt_len (tokens):", trace["prompt_len"])
    print("num steps:", len(trace["steps"]))
    print()
    print("Output (first 300 chars):")
    print(output_text[:300] + ("..." if len(output_text) > 300 else ""))
    print()
    print("=" * 50)
    print("SAMPLE STEP (step 0)")
    print("=" * 50)
    s = trace["steps"][0]
    print("chosen_token:", repr(s["chosen_token"]))
    print("chosen_id:", s["chosen_id"])
    print("logits_topk:", json.dumps(s["logits_topk"], indent=2))
    print("evidence (first 3):", json.dumps(s["evidence"][:3], indent=2))
    print("logit_lens (first 2 layers):")
    for ll in s["logit_lens"][:2]:
        print(f"  layer {ll['layer']}: {ll['topk']}")
    print()
    print("=" * 50)

    out_path = "trace_sample.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(f"Full trace saved to: {out_path}")

if __name__ == "__main__":
    main()
