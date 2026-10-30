"""Gate 1 extension E1 / model-coverage sweep v4 (prereg: icml2027/prereg/sweep_v4_prereg.md). Runs on a GPU pod.
Writes /work/out/sweep/{tag}/{A,B}.jsonl; resumable per (model, cell)."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [  # (tag, hf id, thinking or None)
    ("qwen3-0.6b-nothink", "Qwen/Qwen3-0.6B", False), ("qwen3-0.6b-think", "Qwen/Qwen3-0.6B", True),
    ("qwen3-1.7b-nothink", "Qwen/Qwen3-1.7B", False), ("qwen3-1.7b-think", "Qwen/Qwen3-1.7B", True),
    ("qwen3-4b-nothink", "Qwen/Qwen3-4B", False), ("qwen3-4b-think", "Qwen/Qwen3-4B", True),
    ("qwen3-8b-nothink", "Qwen/Qwen3-8B", False), ("qwen3-8b-think", "Qwen/Qwen3-8B", True),
    ("qwen3-14b-nothink", "Qwen/Qwen3-14B", False), ("qwen3-14b-think", "Qwen/Qwen3-14B", True),
    ("qwen3-32b-think", "Qwen/Qwen3-32B", True),
    ("olmo2-7b", "allenai/OLMo-2-1124-7B-Instruct", None), ("olmo2-13b", "allenai/OLMo-2-1124-13B-Instruct", None),
    ("gemma3-12b", "google/gemma-3-12b-it", None), ("gemma3-27b", "google/gemma-3-27b-it", None),
    ("llama3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/work"); ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--factorial", nargs="*", default=None, help="model tags to run all 6 factorial cells for")
    a = ap.parse_args()
    W = Path(a.work); IN = W / "in"
    ctx = (IN / "prompts" / "context_document_v2.txt").read_text()
    sysp = {c: (IN / "prompts" / f).read_text().replace("{context_document_v2}", ctx).replace("{context_document}", ctx)
            for c, f in [("A", "system_prompt_paid_tier.txt"), ("B", "system_prompt_free_tier_v2.txt")]}
    CELLS = ["A", "B"]
    if a.factorial:
        CELLS = ["A", "A_prime", "A_instr", "B_minus", "B_rnt", "B"]
        sysp = {c: (IN / "prompts" / "factorial" / f"{c}.txt").read_text().replace("{context_document_v2}", ctx) for c in CELLS}
    Q = json.loads((IN / "queries" / "borderline_50.json").read_text())
    for tag, hf, think in MODELS:
        if a.factorial and tag not in a.factorial:
            continue
        out = W / "out" / ("factorial" if a.factorial else "sweep") / tag
        if (out / "DONE").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(hf); tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            if "gemma-3" in hf:
                from transformers import Gemma3ForConditionalGeneration
                model = Gemma3ForConditionalGeneration.from_pretrained(hf, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager").eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
        except Exception as e:
            (out / "ERROR").write_text(repr(e)); print(tag, "load failed", e, flush=True); continue
        kw = {"enable_thinking": think} if think is not None else {}
        max_new = 2048 if think else 512
        for cell in CELLS:
            items = [(q, n) for n in range(a.runs) for q in Q]
            rows = []
            for i in range(0, len(items), a.batch):
                ch = items[i:i + a.batch]
                texts = [tok.apply_chat_template([{"role": "system", "content": sysp[cell]}, {"role": "user", "content": q["query"]}],
                                                 tokenize=False, add_generation_prompt=True, **kw) for q, _ in ch]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9,
                                       pad_token_id=tok.pad_token_id)
                dec = tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                rows += [dict(model=tag, hf=hf, thinking=think, cell=cell, query_id=q["id"], query=q["query"], run=n, response=r)
                         for (q, n), r in zip(ch, dec)]
            (out / f"{cell}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out / "DONE").write_text(f"{time.time() - t0:.0f}s")
        print(f"{tag}: done in {time.time() - t0:.0f}s", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
