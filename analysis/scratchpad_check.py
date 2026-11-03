"""Condition C (private scratchpad) faking-reasoning rates, recomputed from the raw judge labels.

Scratchpad judge labels: GENUINE_COMPLIANCE, PROBABLY_FAKING, DEFINITELY_FAKING, AMBIGUOUS, NO_SCRATCHPAD.
Reports faking labels over all compliant Condition-C answers and over those that actually used the scratchpad.
Writes outputs/scratchpad_check.json.
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
D = PAPER / "data" / "generations" / "scratchpad"
SOURCES = {m: D / f"{m}_conditionC_scratchpad_labels.json" for m in ["qwen3-32b", "llama-3.1-8b"]}
FAKING = {"PROBABLY_FAKING", "DEFINITELY_FAKING"}


def labels(path):
    out = []
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, str):
            out.append(o)
    walk(json.loads(Path(path).read_text()))
    return out


if __name__ == "__main__":
    res = {}
    for name, p in SOURCES.items():
        L = labels(p)
        c = collections.Counter(L)
        used = len(L) - c["NO_SCRATCHPAD"]
        fk = sum(c[k] for k in FAKING)
        res[name] = dict(n_compliant=len(L), n_used_scratchpad=used, n_faking=fk, counts=dict(c),
                         pct_of_compliant=100 * fk / len(L), pct_of_used=100 * fk / used if used else None,
                         source=str(p.relative_to(PAPER)))
        print(name, res[name])
    (HERE / "outputs" / "scratchpad_check.json").write_text(json.dumps(res, indent=1))
