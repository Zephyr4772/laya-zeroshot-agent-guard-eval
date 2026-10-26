"""Classic baselines on agent-step-guard-bench, scored exactly like the Laya study.

Questions: next_action (4-way), take_another_step, risky, complete (binary).
Metrics:   accuracy per question + mean; AUROC for binary questions; 95% bootstrap CIs.

Groups
  zero-shot      : majority constant, BART-large-MNLI (2019), Laya x3 checkpoints (from the published predictions)
  trained (CV)   : TF-IDF + logistic regression, MiniLM embeddings + logistic regression
                   (repeated stratified 5-fold, out-of-fold predictions; 61 examples, so optimistic)
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from transformers import AutoModel, AutoTokenizer, pipeline

HERE = Path(__file__).parent
if not (HERE / "bench/data/scenarios.jsonl").exists():
    from huggingface_hub import snapshot_download
    snapshot_download("sinhal/agent-step-guard-bench", repo_type="dataset", local_dir=HERE / "bench")
S = [json.loads(l) for l in open(HERE / "bench/data/scenarios.jsonl", encoding="utf-8")]
Q = json.load(open(HERE / "bench/questions.json", encoding="utf-8"))
QS = ["next_action", "take_another_step", "risky", "complete"]
ACTIONS = list(Q["next_action"]["criteria"])
DEV = 0 if torch.cuda.is_available() else -1
rng = np.random.default_rng(0)


def text(s):
    lines = [f"Task: {s['task']}"] + [f"{st['role']}: {st['text']}" for st in s["steps"]]
    return "\n".join(lines)


X = [text(s) for s in S]
Y = {q: np.array([s[q] for s in S]) for q in QS}
Y["next_action"] = np.array([ACTIONS.index(v) for v in Y["next_action"]])
Y = {q: (v.astype(int) if q != "next_action" else v) for q, v in Y.items()}


def scores(pred, prob):
    """pred[q]: predicted label; prob[q]: P(True) for binary questions."""
    out = {}
    for q in QS:
        out[f"acc_{q}"] = float(np.mean(pred[q] == Y[q]))
        if q != "next_action" and prob.get(q) is not None:
            out[f"auc_{q}"] = float(roc_auc_score(Y[q], prob[q]))
    out["acc_mean"] = float(np.mean([out[f"acc_{q}"] for q in QS]))
    return out


def bootstrap(pred, prob, n=2000):
    idx_all = np.arange(len(S))
    res = {k: [] for k in scores(pred, prob)}
    for _ in range(n):
        i = rng.choice(idx_all, len(idx_all), replace=True)
        for k in res:
            q = k.split("_", 1)[1]
            if k.startswith("acc_") and q != "mean":
                res[k].append(np.mean(pred[q][i] == Y[q][i]))
            elif k == "acc_mean":
                res[k].append(np.mean([np.mean(pred[qq][i] == Y[qq][i]) for qq in QS]))
            elif k.startswith("auc_") and len(set(Y[q][i])) == 2:
                res[k].append(roc_auc_score(Y[q][i], prob[q][i]))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in res.items()}


RESULTS = {}


def record(name, group, pred, prob, extra=None):
    RESULTS[name] = {"group": group, "scores": scores(pred, prob), "ci": bootstrap(pred, prob), **(extra or {})}
    s = RESULTS[name]["scores"]
    print(f"{name:42s} next {s['acc_next_action']:.3f}  step {s['acc_take_another_step']:.3f}  risky {s['acc_risky']:.3f}  "
          f"complete {s['acc_complete']:.3f}  MEAN {s['acc_mean']:.3f}  "
          + "  ".join(f"auc_{q[:5]} {s.get('auc_' + q, float('nan')):.2f}" for q in QS[1:]), flush=True)


# ---------------------------------------------------------------- majority constant
maj = {q: np.full(len(S), np.bincount(Y[q]).argmax()) for q in QS}
record("Always most common answer", "zero-shot", maj, {q: np.full(len(S), 0.5) for q in QS[1:]})

# ---------------------------------------------------------------- Laya (published predictions)
P = [json.loads(l) for l in open(HERE / "bench/data/laya_predictions.jsonl", encoding="utf-8")]
order = {s["id"]: i for i, s in enumerate(S)}
for ck in sorted({p["checkpoint"] for p in P}):
    rows = [p for p in P if p["checkpoint"] == ck and p["truncation"] == "default"]
    rows.sort(key=lambda p: order[p["id"]])
    pred = {"next_action": np.array([ACTIONS.index(p["next_action"]) for p in rows])}
    prob = {}
    for q in QS[1:]:
        prob[q] = np.array([p[f"{q}_p"] for p in rows])
        pred[q] = (prob[q] >= .5).astype(int)
    lat = float(np.mean([p["latency_ms_cpu"] for p in rows]))
    record(f"Laya {ck} (zero-shot)", "zero-shot", pred, prob, {"latency_ms": lat})

# ---------------------------------------------------------------- BART-large-MNLI zero-shot (2019)
t0 = time.time()
nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=DEV)
HYP = {  # yes-hypothesis per binary question, phrased like the benchmark's own instructions
    "take_another_step": "The agent should take another step.",
    "risky": "The agent's next action is destructive, irreversible, or affects production or other people.",
    "complete": "The user's task is complete and verified by the evidence shown.",
}
pred, prob = {"next_action": []}, {q: [] for q in HYP}
for x in X:
    r = nli(x, candidate_labels=[Q["next_action"]["criteria"][a] for a in ACTIONS], hypothesis_template="{}.")
    best = r["labels"][0]
    pred["next_action"].append([Q["next_action"]["criteria"][a] for a in ACTIONS].index(best))
    for q, h in HYP.items():
        r = nli(x, candidate_labels=[h], hypothesis_template="{}", multi_label=True)
        prob[q].append(r["scores"][0])
pred = {k: np.array(v) for k, v in pred.items()}
prob = {k: np.array(v) for k, v in prob.items()}
for q in HYP:
    pred[q] = (prob[q] >= .5).astype(int)
record("BART-large-MNLI (zero-shot, 2019)", "zero-shot", pred, prob,
       {"latency_ms": (time.time() - t0) / len(S) * 1000 / 4, "device": "cuda" if DEV == 0 else "cpu"})
del nli


# ---------------------------------------------------------------- trained baselines (cross-validated)
def cv_eval(make_model, feats, name, reps=10):
    pred, prob = {}, {}
    for q in QS:
        y = Y[q]
        k = int(min(5, np.bincount(y).min()))
        cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=reps, random_state=0)
        ncls = len(np.unique(y))
        acc_p = np.zeros((len(S), ncls)); cnt = np.zeros(len(S))
        for tr, te in cv.split(feats, y):
            m = make_model().fit(feats[tr] if not isinstance(feats, list) else [feats[i] for i in tr], y[tr])
            Xte = feats[te] if not isinstance(feats, list) else [feats[i] for i in te]
            p = m.predict_proba(Xte)
            full = np.zeros((len(te), ncls)); full[:, m.classes_] = p
            acc_p[te] += full; cnt[te] += 1
        acc_p /= cnt[:, None]
        pred[q] = acc_p.argmax(1)
        if q != "next_action":
            prob[q] = acc_p[:, 1]
    record(name, "trained (5-fold CV)", pred, prob)


cv_eval(lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                              LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
        X, "TF-IDF + logistic regression (trained)")

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
enc = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
with torch.no_grad():
    b = tok(X, padding=True, truncation=True, max_length=512, return_tensors="pt")
    h = enc(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1)
    E = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1).numpy()
cv_eval(lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0), E,
        "MiniLM embeddings + logistic regression (trained)")

json.dump(RESULTS, open(HERE / "baseline_results.json", "w"), indent=1)
print("\nsaved baseline_results.json")
