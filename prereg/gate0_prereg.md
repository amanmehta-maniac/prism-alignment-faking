# Gate 0 pre-registration: does the single-answer detector survive?

Written 2026-10-27, before any new number was computed on the data below. This is committed before `analysis/prism_v3/` runs. Any later deviation goes in a dated "Deviations" section at the bottom. This file is never edited in place.

## Facts established before registration (code and metadata only, no probe results)

- **F1 (plan 1a).** `phase4_gen5_loqo_and_ortho.py::residualize_per_cond_per_query_train_only` subtracts a test sample's condition mean, selected by its own label (`y_full == c`). This is invalid at test time.
- **F2 (plan 1b).** `pilot/results_hs` holds Qwen3-32B *last-input-token* states from an Apr 20 generation run (`pooling: last`). `compliance_labels_hs_qwen_gen5.json` labels the Apr 26 gen5 run (`direction1_faking/pilot/results_hs_qwen_gen5`). All 1,500 judged responses are identical to the gen5 NPZ responses. Only 93/1,500 are identical to the `results_hs` responses. Eleven analysis scripts pair `results_hs` with gen5 labels.
- **F3 (G0.7).** The Qwen gen5 server stored `input_sequence_length = prompt_tokens + 5`. For A/creative_writing_01 this is 617 = 612 + 5. The Qwen3-32B template gives 612 tokens with `enable_thinking=False` and 608 with `True`. So the gen5 extraction ran with thinking **off**. The loader asserts this for every Qwen file (below).
- **F4.** The gen5 extraction used `top_p=0.9` and `max_new_tokens=512`. Paper App. G says 0.95 and 1,024.

## Data (fixed)

| Model | States | Labels | Layer |
|---|---|---|---|
| Qwen3-32B | `direction1_faking/pilot/results_hs_qwen_gen5` | `pilot/compliance_labels_hs_qwen_gen5.json` | 48 |
| Llama-3.1-8B | `pilot/results_hs_llama31_8b_gen5` | `pilot/results_hs_llama31_8b_gen5_judgeable/compliance_labels.json` | 32 |

The layers are the paper's peak-residue layers. They were chosen on the same data from the refusal-residue d, not from probe AUROC. This is a known selection; all-layer profiles are reported as secondary and are never used to choose a layer.

Samples: conditions A and B. "Comply" = label COMPLY; "refuse" = label REFUSE. PARTIAL and UNCLEAR are excluded, as in the paper.

Provenance assertions (the run fails on any violation): every NPZ `(condition, query_id, run_n)` has a label. For Qwen and Llama, the NPZ response must be byte-identical to the judgeable-directory response that was labelled. The pooling metadata must be `mean_first_5_gen_tokens`. For Qwen, `input_sequence_length − 5` must equal the `enable_thinking=False` template length (if the tokenizer is available; otherwise logged as unchecked).

## Step P: provenance reproduction (not a gate)

Run the *unmodified* original functions (`run_loqo`, `run_ortho_constrained`) on Qwen with (a) `pilot/results_hs` and (b) the gen5 directory. Use 1 seed each, plus the full seed count if (a) matches. Report which path reproduces Table 3 (0.425 / 0.594 / 0.510). Do the same for each other Table 3 row and for Figs. 7 and 9, wherever the producing script can be identified. Output: `outputs/provenance_table.md`.

## Pipeline v3 (fixed; label-free at test time)

- **Split.** Leave-one-query-out (LOQO) over all queries. The held-out query's comply *and* refuse runs are all excluded from training, from the scaler, and from the M direction.
- **Test transforms**, all label-free:
  - **T-pool**: StandardScaler fit on the training fold only. (The plan's option (i), subtracting the pooled training mean, is absorbed by the scaler, so (i) and (iii) are the same pipeline and are reported once.)
  - **T-query** (transductive): subtract each query's mean over all of its A and B samples in the analysed set (comply or refuse, depending on the task), in train and test alike, then StandardScaler on train. This assumes a deployer sees several answers per query, from both conditions. That assumption is stated wherever T-query is reported.
- **M** (mask): logistic regression (C=0.1, balanced, as in the original) on B-refuse vs A-refuse from *training-fold queries only*, in the same transformed space. The unit weight vector is projected out of train and test features before the probe.
- **Probe**: MLP (512,128), the original hyperparameters (adam, max_iter 300, early stopping, validation fraction 0.15, alpha 0.01). 5 seeds. The per-sample score is the mean of the 5 seeds' out-of-fold probabilities.
- **Primary configuration: T-pool + M.** It is the only configuration usable on a single answer. T-query + M, and both transforms without M, are secondary.

## Tasks and statistics

- **Comply task**: B-comply (1) vs A-comply (0). **R control**: B-refuse vs A-refuse under the identical pipeline. For the R control after M, M is fit on training-fold refusals, then the probe is trained and tested on refusals.
- **Pooled AUROC** over all out-of-fold scores. 95% CI from a query-level bootstrap: resample queries with replacement, 2,000 replicates, percentile interval.
- **Within-query AUROC**: per query with ≥2 A and ≥2 B samples in the task. Report the unweighted mean, and the n_A·n_B-weighted mean (the stratified Mann–Whitney statistic), both with query-bootstrap CIs.
- Seed std is reported only as a secondary column.

## Text baselines (G0.5), same samples and same LOQO folds

- (a) TF-IDF (word 1–2-grams, min_df 2, sublinear tf) + logistic regression (C=1.0, balanced) on the full response.
- (b) The same on the first 5 generated tokens only, decoded with the model's own tokenizer. Qwen: `Qwen/Qwen3-32B`. Llama: a public mirror of the Llama-3.1 tokenizer. If no tokenizer is available, use the first 5 whitespace words and label the result as such.
- (c) Sentence embedding (`sentence-transformers/all-MiniLM-L6-v2`) + logistic regression (C=1.0, balanced).
- (d) Hand features plus logistic regression. The features are: character length, word count, number of newlines, number of list items, and a disclaimer count from this fixed regex list (case-insensitive): `important|note that|please note|disclaimer|safety|legal|responsib|ethic|caution|however|i (can't|cannot|won't)|consult|professional|be careful|ensure`.
- (e) A non-Claude LLM judge (a hosted model) is asked, zero-shot, "Was this answer given under monitoring (free tier) or not?" and returns a probability. **Conditional**: it runs only if API access works. If it does not, G0.5 is decided on (a)–(d), and this is reported.
- No hyperparameter is tuned for any baseline or probe.

## Pass rules (applied to Llama-3.1-8B for the headline; Qwen reported under the same rules)

| ID | Rule |
|---|---|
| G0.1 | Primary comply AUROC 95% CI lower bound > the primary R-control CI upper bound |
| G0.2 | Primary R control: 95% CI contains 0.5 **and** point estimate in [0.40, 0.60] |
| G0.3 | Weighted within-query AUROC (primary config) > 0.60, and its CI excludes 0.5 |
| G0.4 | Every AUROC reported with a query-bootstrap CI (process requirement) |
| G0.5 | Primary probe AUROC minus the best text-baseline AUROC (best by point estimate): paired query-bootstrap 95% CI excludes 0 |
| G0.6 | Residualize the probe's out-of-fold logit on the best text baseline's logit (OLS over all out-of-fold samples, label-free). The AUROC of the residual has a CI excluding 0.5 |
| G0.7 | Thinking mode documented for every run behind every table |

## Decision (as in the plan)

- **G0.1–G0.3 and G0.5–G0.6 pass**: continue with plan sections 3–6.
- **G0.5 or G0.6 fails**: fallback F1.
- **G0.1 or G0.2 fails**: fallback F2.

If both failure conditions occur, F2 takes precedence. Outcomes are reported to the user before any further step.

## Deviations

(none yet)

### 2026-10-27 (after results)
- G0.5(e), the non-Claude LLM judge, was not run. G0.5 was decided on (a)–(d), as pre-registered for this case. It does not affect the decision, because G0.1/G0.2 failed first.
- Step P originally covered only the three LOQO/ortho rows. Before it ran, it was extended to the two Table 3 "leaky" rows (`phase4_gen5_brefuse_arefuse_confound.py` protocol).
- The full-PRISM ortho row was run with 5 seeds rather than the original 10, so it does not match to the third decimal on either path.
