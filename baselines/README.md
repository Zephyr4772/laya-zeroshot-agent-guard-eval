# Classic baselines

Compares Laya against classifiers from 2019 and earlier on [agent-step-guard-bench](https://huggingface.co/datasets/sinhal/agent-step-guard-bench), scored the same way as the main study (accuracy per question, AUROC for yes/no questions, 95% bootstrap CIs).

```bash
python baselines.py   # majority, Laya x3 (published predictions), BART-large-MNLI zero-shot, TF-IDF+LR, MiniLM+LR
python loco.py        # leave-one-category-out check for the trained baselines + CI summary
```

Both scripts download the benchmark on first run. BART-large-MNLI is about 1.6 GB.

| System | Training | Mean accuracy [95% CI] |
|---|---|---|
| Always most common answer | none | 67.6% [64.3, 70.5] |
| BART-large-MNLI (2019) | zero-shot | 49.2% [43.0, 54.9] |
| Laya typed-decisions | zero-shot | 52.5% [45.9, 58.6] |
| TF-IDF + logistic regression | random 5-fold CV | 73.0% [65.6, 79.5] |
| TF-IDF + logistic regression | leave one category out | 56.1% |
| MiniLM + logistic regression | random 5-fold CV | 66.8% [59.0, 74.2] |
| MiniLM + logistic regression | leave one category out | 55.3% |

The trained models' random-fold scores are inflated: within a category the answer is nearly constant, so they memorise templates. Held-out categories show the real picture. Full results: `baseline_results.json`, `loco_results.json`.
