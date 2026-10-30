"""Gate 0 text-only baselines (G0.5), same samples and LOQO folds as the probe.

(a) tfidf_full   TF-IDF 1-2gram + logistic on the full response
(b) tfidf_first5 same on the first 5 tokens (model's own tokenizer)
(c) minilm       all-MiniLM-L6-v2 embedding + logistic
(d) handfeat     length / format / disclaimer-count features + logistic
Out-of-fold scores saved to outputs/scores/{model}_{task}_text_{name}.npz.
"""
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common

SCORES = common.OUT / "scores"
C = common.CFG["text_logreg_C"]
DISC = re.compile(common.CFG["disclaimer_regex"], re.I)


def first5(responses, tok_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_name)
    return [tok.convert_tokens_to_string(tok.tokenize(r)[:5]) for r in responses]


def handfeat(responses):
    return np.array([[len(r), len(r.split()), r.count("\n"),
                      len(re.findall(r"^\s*(?:[-*•]|\d+\.)\s", r, re.M)),
                      len(DISC.findall(r))] for r in responses], dtype=float)


def loqo_text(features, y, q, kind):
    """kind='text' -> features is a list of str (TF-IDF fit per fold); else a matrix."""
    s = np.full(len(y), np.nan)
    for held in np.unique(q):
        tr, te = q != held, q == held
        if kind == "text":
            model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                  LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
            ftr = [features[i] for i in np.where(tr)[0]]
            fte = [features[i] for i in np.where(te)[0]]
        else:
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
            ftr, fte = features[tr], features[te]
        s[te] = model.fit(ftr, y[tr]).predict_proba(fte)[:, 1]
    return s


def main(models):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    SCORES.mkdir(parents=True, exist_ok=True)
    for key in models:
        d = common.load_model(key)
        L = common.CFG["models"][key]["peak_layer"]
        for tname, label in [("comply", "COMPLY"), ("refuse", "REFUSE")]:
            X, y, q, resp, idx = common.task(d, L, label)
            resp = list(resp)
            feats = {
                "tfidf_full": (resp, "text"),
                "tfidf_first5": (first5(resp, common.CFG["models"][key]["tokenizer"]), "text"),
                "minilm": (st.encode(resp, batch_size=64, show_progress_bar=False), "mat"),
                "handfeat": (handfeat(resp), "mat"),
            }
            for name, (f, kind) in feats.items():
                s = loqo_text(f, y, q, kind)
                np.savez(SCORES / f"{key}_{tname}_text_{name}.npz", S=s[None], y=y, q=q, idx=idx,
                         commit=common.git_commit())
                print(f"{key} {tname} {name}: AUROC={common.auroc(y, s):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["llama", "qwen"])
