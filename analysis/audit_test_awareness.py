"""§7.3 audit: PRISM-style checks on Abdelnabi & Salem (2025), "Linear Control of Test Awareness"
(github.com/microsoft/Test_Awareness_Steering), Qwen3-32B trigger data. CPU only; their code is NOT
executed. Their get_classes/split_train_test logic is reimplemented from the source (same seeds).

Checks:
  (1) duplicates: entries appended once per positive key, so double-positive traces appear twice
  (2) pair leakage: real/hypothetical halves of one trigger split independently across train/test
  (3) prompt-type shortcut: how well "is the prompt hypothetical?" alone predicts the label
  (4) text baseline: TF-IDF + logistic regression on the evidence sentences the probe is read from,
      on their exact split, and on a pair-grouped split
Usage: python -I audit_test_awareness.py /path/to/triggers_qwen3_with_GPT_labels_evidence.json
"""
import json
import random
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

POS = ["Yes", "Maybe"]
KEYS = ["model_awareness", "recognition"]


def flatten(data):
    out = []
    for i, e in enumerate(data):
        for pk in ["real", "hypothetical"]:
            r = {"pair": i, "prompt_type": pk, "prompt": e[pk]}
            for k in KEYS:
                v = e.get(f"{pk}_{k}", {}) or {}
                r[k] = v.get("conclusion"); r[k + "_evidence"] = v.get("evidence", [])
            out.append(r)
    return out


def get_classes(entries):
    pos, neg = [], []
    for e in entries:
        for k in KEYS:                                  # appended once per positive key (as in source)
            if e[k] in POS:
                pos.append(e)
        if e["model_awareness"] not in POS and e["recognition"] not in POS and e["model_awareness"] and e["recognition"]:
            neg.append(e)
    random.Random(4).shuffle(neg); random.Random(4).shuffle(pos)
    neg = neg[: len(pos)]
    np.random.seed(0)
    t = int(min(len(pos), len(neg)) / 3)
    pt = set(np.random.permutation(len(pos))[:t])
    np.random.seed(0)
    nt = set(np.random.permutation(len(neg))[:t])
    return ([p for i, p in enumerate(pos) if i not in pt], [p for i, p in enumerate(pos) if i in pt],
            [n for i, n in enumerate(neg) if i not in nt], [n for i, n in enumerate(neg) if i in nt])


def evidence_text(e, positive):
    keys = [k for k in KEYS if e[k] in POS] if positive else KEYS
    return " ".join(s for k in keys for s in e[k + "_evidence"]).lower()


def main(path):
    data = json.load(open(path))
    ptr, pte, ntr, nte = get_classes(flatten(data))
    rep = {"n_pairs": len(data), "train": [len(ptr), len(ntr)], "test": [len(pte), len(nte)]}
    key = lambda e: (e["pair"], e["prompt_type"])
    trk = {key(e) for e in ptr + ntr}
    rep["dup_positive_total"] = len(ptr + pte) - len({key(e) for e in ptr + pte})
    rep["test_pos_exact_in_train"] = sum(key(e) in {key(x) for x in ptr} for e in pte)
    rep["test_pair_partner_in_train"] = sum(any((e["pair"], o) in trk for o in ["real", "hypothetical"] if o != e["prompt_type"])
                                            for e in pte + nte)
    rep["test_n"] = len(pte + nte)
    # (3) prompt-type shortcut
    tr = ptr + ntr; te = pte + nte
    ytr = np.array([1] * len(ptr) + [0] * len(ntr)); yte = np.array([1] * len(pte) + [0] * len(nte))
    hyp = lambda S: np.array([e["prompt_type"] == "hypothetical" for e in S], float)
    rep["frac_hypothetical"] = {"pos": float(hyp(ptr + pte).mean()), "neg": float(hyp(ntr + nte).mean())}
    rep["prompt_type_only_test_acc"] = float(accuracy_score(yte, hyp(te) > 0.5))
    rep["prompt_type_only_test_auroc"] = float(roc_auc_score(yte, hyp(te)))
    # (4) text baseline on evidence sentences, their split
    Ttr = [evidence_text(e, y == 1) for e, y in zip(tr, ytr)]; Tte = [evidence_text(e, y == 1) for e, y in zip(te, yte)]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(Ttr)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(vec.transform(Ttr), ytr)
    p = clf.predict_proba(vec.transform(Tte))[:, 1]
    rep["text_evidence_their_split"] = {"acc": float(accuracy_score(yte, p > 0.5)), "auroc": float(roc_auc_score(yte, p))}
    # pair-grouped 5-fold (deduplicated) as the PRISM-clean reference
    allx = {key(e): (e, 1) for e in ptr + pte}
    allx.update({key(e): (e, 0) for e in ntr + nte if key(e) not in allx})
    items = list(allx.values())
    T = np.array([evidence_text(e, y == 1) for e, y in items], dtype=object); Y = np.array([y for _, y in items])
    G = np.array([e["pair"] for e, _ in items]); s = np.zeros(len(Y))
    for a, b in GroupKFold(5).split(T, Y, G):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(T[a])
        s[b] = LogisticRegression(C=1.0, max_iter=2000).fit(v.transform(T[a]), Y[a]).predict_proba(v.transform(T[b]))[:, 1]
    rep["text_evidence_pair_grouped"] = {"acc": float(accuracy_score(Y, s > 0.5)), "auroc": float(roc_auc_score(Y, s)), "n": int(len(Y))}
    top = np.argsort(clf.coef_.ravel())
    names = np.array(vec.get_feature_names_out())
    rep["top_positive_ngrams"] = names[top[-15:]][::-1].tolist()
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    out = main(sys.argv[1])
    json.dump(out, open(sys.argv[2], "w"), indent=1) if len(sys.argv) > 2 else None
