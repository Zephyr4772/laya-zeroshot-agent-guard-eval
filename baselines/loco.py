"""Robustness check: leave-one-category-out for the trained baselines, plus CIs for the headline table."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).parent
if not (HERE / "bench/data/scenarios.jsonl").exists():
    from huggingface_hub import snapshot_download
    snapshot_download("sinhal/agent-step-guard-bench", repo_type="dataset", local_dir=HERE / "bench")
S = [json.loads(l) for l in open(HERE / "bench/data/scenarios.jsonl", encoding="utf-8")]
Q = json.load(open(HERE / "bench/questions.json", encoding="utf-8"))
QS = ["next_action", "take_another_step", "risky", "complete"]
ACTIONS = list(Q["next_action"]["criteria"])
X = ["\n".join([f"Task: {s['task']}"] + [f"{st['role']}: {st['text']}" for st in s["steps"]]) for s in S]
cats = np.array([s["category"] for s in S])
Y = {q: np.array([ACTIONS.index(s[q]) if q == "next_action" else int(s[q]) for s in S]) for q in QS}

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
with torch.no_grad():
    b = tok(X, padding=True, truncation=True, max_length=512, return_tensors="pt")
    h = enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1)
    E = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1).numpy()

models = {
    "TF-IDF + LR": (lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                          LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)), X),
    "MiniLM + LR": (lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0), E),
}
out = {}
for name, (mk, F) in models.items():
    res = {}
    for q in QS:
        y = Y[q]; ncls = len(ACTIONS) if q == "next_action" else 2
        P = np.zeros((len(S), ncls))
        for c in np.unique(cats):
            te = np.where(cats == c)[0]; tr = np.where(cats != c)[0]
            if len(np.unique(y[tr])) < 2:
                P[te] = np.bincount(y[tr], minlength=ncls) / len(tr); continue
            mdl = mk().fit([F[i] for i in tr] if isinstance(F, list) else F[tr], y[tr])
            p = mdl.predict_proba([F[i] for i in te] if isinstance(F, list) else F[te])
            P[te][:, :] = 0
            full = np.zeros((len(te), ncls)); full[:, mdl.classes_] = p; P[te] = full
        res[f"acc_{q}"] = float(np.mean(P.argmax(1) == y))
        if q != "next_action":
            res[f"auc_{q}"] = float(roc_auc_score(y, P[:, 1]))
    res["acc_mean"] = float(np.mean([res[f"acc_{q}"] for q in QS]))
    out[name] = res
    print(f"LOCO {name:12s} next {res['acc_next_action']:.3f} step {res['acc_take_another_step']:.3f} risky {res['acc_risky']:.3f} "
          f"complete {res['acc_complete']:.3f} MEAN {res['acc_mean']:.3f} | auc step {res['auc_take_another_step']:.2f} "
          f"risky {res['auc_risky']:.2f} complete {res['auc_complete']:.2f}")
json.dump(out, open(HERE / "loco_results.json", "w"), indent=1)

R = json.load(open(HERE / "baseline_results.json"))
print("\nheadline 95% CIs (bootstrap):")
for k, v in R.items():
    c = v["ci"]
    print(f"{k:50s} mean acc {v['scores']['acc_mean']:.3f} [{c['acc_mean'][0]:.3f}, {c['acc_mean'][1]:.3f}]  "
          f"risky auc {v['scores'].get('auc_risky', float('nan')):.2f} [{c.get('auc_risky', (float('nan'),)*2)[0]:.2f}, {c.get('auc_risky', (float('nan'),)*2)[1]:.2f}]")
