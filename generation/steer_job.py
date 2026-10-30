"""Gate 2 steering generations (runs on a GPU pod). Prereg: icml2027/prereg/gate2_steering_prereg.md.

python steer_job.py --model meta-llama/Llama-3.1-8B-Instruct --key llama --work /work
Expects /work/in/{key}_directions.npz, directions_meta.json, prompts/*.txt, queries/*.json.
Writes /work/out/{key}/{arm}.jsonl (resumable: completed (qid, run) pairs are skipped).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PEAK_MID = {"llama": (32, 16), "qwen": (48, 32)}
BETAS = [0.5, 1, 2, 4]


def arms(key, which="main"):
    pk, mid = PEAK_MID[key]
    if which == "d3":   # matched additive control at the peak layer (deviation D3)
        return [(f"D3_{c}_{'sub' if c == 'B' else 'add'}_condref_L{pk}_b{b}", c, "borderline", 5,
                 ("add", "v_cond_refuse", -b if c == "B" else b, pk)) for c in ("B", "A") for b in (1, 2, 4)]
    A = [("E1a_A_base", "A", "harmbench", 2, None), ("E1a_A_abl_refusal", "A", "harmbench", 2, ("ablate", "v_refusal")),
         ("E1b_A_base", "A", "borderline", 4, None), ("E1b_A_abl_refusal", "A", "borderline", 4, ("ablate", "v_refusal")),
         ("E2a_A_base", "A", "borderline", 5, None),
         ("E2_B_base", "B", "borderline", 5, None), ("E2_B_abl_faking", "B", "borderline", 5, ("ablate", "v_faking")),
         ("E2_B_abl_condref", "B", "borderline", 5, ("ablate", "v_cond_refuse"))]
    for l in (pk, mid):
        for b in BETAS:
            A.append((f"E3_B_sub_L{l}_b{b}", "B", "borderline", 5, ("add", "v_faking", -b, l)))
            A.append((f"E3a_A_add_L{l}_b{b}", "A", "borderline", 5, ("add", "v_faking", b, l)))
    return A


class Hooks:
    def __init__(self, model, dirs, gap):
        # hidden_states[l] = output of block l-1 for l < L; hidden_states[L] is AFTER the final norm
        self.layers = list(model.model.layers[:-1]) + [model.model.norm]
        self.dirs = {k: torch.tensor(v, dtype=model.dtype, device=model.device) for k, v in dirs.items()}
        self.gap, self.handles = gap, []

    def set(self, spec):
        for h in self.handles:
            h.remove()
        self.handles = []
        if spec is None:
            return
        if spec[0] == "ablate":
            V = self.dirs[spec[1]]
            for i, layer in enumerate(self.layers):
                v = V[i + 1]
                def hook(mod, inp, out, v=v):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h - (h @ v).unsqueeze(-1) * v
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                self.handles.append(layer.register_forward_hook(hook))
        else:
            _, name, beta, l = spec
            delta = beta * self.gap[l] * self.dirs[name][l]
            def hook(mod, inp, out, delta=delta):
                h = out[0] if isinstance(out, tuple) else out
                h = h + delta
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.handles.append(self.layers[l - 1].register_forward_hook(hook))  # l = L hooks the final norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--work", default="/work"); ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--arms", default="main")
    a = ap.parse_args()
    W = Path(a.work); IN = W / "in"; OUT = W / "out" / a.key; OUT.mkdir(parents=True, exist_ok=True)
    ctx = (IN / "prompts" / "context_document_v2.txt").read_text()
    sysp = {c: (IN / "prompts" / f).read_text().replace("{context_document_v2}", ctx).replace("{context_document}", ctx)
            for c, f in [("A", "system_prompt_paid_tier.txt"), ("B", "system_prompt_free_tier_v2.txt")]}
    Q = {"borderline": json.loads((IN / "queries" / "borderline_50.json").read_text()),
         "harmbench": json.loads((IN / "queries" / "harmbench_50.json").read_text())}
    z = np.load(IN / f"{a.key}_directions.npz")
    gap = json.loads((IN / "directions_meta.json").read_text())[a.key]["gap"]
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    hooks = Hooks(model, {k: z[k] for k in z.files}, gap)
    kw = {"enable_thinking": False} if a.key == "qwen" else {}
    for arm, cond, qset, runs, spec in arms(a.key, a.arms):
        f = OUT / f"{arm}.jsonl"
        done = {(r["query_id"], r["run"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else set()
        todo = [(q, n) for n in range(runs) for q in Q[qset] if (q["id"], n) not in done]
        if not todo:
            continue
        hooks.set(spec); t0 = time.time()
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            texts = [tok.apply_chat_template([{"role": "system", "content": sysp[cond]}, {"role": "user", "content": q["query"]}],
                                             tokenize=False, add_generation_prompt=True, **kw) for q, _ in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9,
                                     pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            with f.open("a") as fh:
                for (q, n), r in zip(chunk, dec):
                    fh.write(json.dumps(dict(arm=arm, cond=cond, qset=qset, query_id=q["id"], query=q["query"],
                                             run=n, spec=str(spec), response=r)) + "\n")
        hooks.set(None)
        print(f"{arm}: {len(todo)} in {time.time() - t0:.0f}s", flush=True)
    (OUT / ("DONE" if a.arms == "main" else f"DONE_{a.arms}")).write_text(time.strftime("%c"))


if __name__ == "__main__":
    main()
