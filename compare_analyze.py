"""Score results/compare_julia.json + results/compare_laya.json (identical plain-text input). Prints Markdown tables."""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
runs = {}
for f in ("compare_julia.json", "compare_laya.json"):
    runs.update(json.load(open(os.path.join(HERE, "results", f))))
QS = ["next_action", "take_another_step", "risky", "complete"]
ORDER = ["julia-1", "laya-typed-decisions", "laya-multilingual", "laya-english"]
golds = [r["gold"] for r in runs["julia-1"]]
cats = list(dict.fromkeys(r["cat"] for r in runs["julia-1"]))


def pred(a, q):
    return a[q]["choice"] if a[q]["type"] == "choice" else a[q]["noul"] >= 0.5


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def auroc_ci(scores, labels, n=2000):
    rng, idx, out = random.Random(0), list(range(len(scores))), []
    for _ in range(n):
        b = [rng.choice(idx) for _ in idx]
        lb = [labels[i] for i in b]
        if any(lb) and not all(lb):
            out.append(auroc([scores[i] for i in b], lb))
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


maj = {q: Counter(g[q] for g in golds).most_common(1)[0][0] for q in QS}
print("### Accuracy by question (%)\n")
print("| system | next_action | take_another_step | risky | complete | mean |")
print("|---|---|---|---|---|---|")
accs = [sum(maj[q] == g[q] for g in golds) / len(golds) for q in QS]
print("| always the most common answer | %s | %.1f |" % (" | ".join("%.1f" % (100 * a) for a in accs), 100 * sum(accs) / 4))
for m in ORDER:
    rows = runs[m]
    accs = [sum(pred(r["answers"], q) == r["gold"][q] for r in rows) / len(rows) for q in QS]
    print("| %s | %s | %.1f |" % (m, " | ".join("%.1f" % (100 * a) for a in accs), 100 * sum(accs) / 4))

print("\n### next_action accuracy by category (%)\n")
print("| system | " + " | ".join(cats) + " |")
print("|---|" + "---|" * len(cats))
for m in ORDER:
    rows = runs[m]
    print("| %s | %s |" % (m, " | ".join("%.0f" % (100 * sum(pred(r["answers"], "next_action") == r["gold"]["next_action"] for r in rows if r["cat"] == c) / sum(r["cat"] == c for r in rows)) for c in cats)))

print("\n### What each model answers for next_action (gold: %s)\n" % dict(Counter(g["next_action"] for g in golds)))
for m in ORDER:
    print("- %s: %s" % (m, dict(Counter(pred(r["answers"], "next_action") for r in runs[m]))))

print("\n### Yes/no questions: how often 'yes', and AUROC with 95% CI\n")
print("| question (true rate) | system | says yes | AUROC [95% CI] |")
print("|---|---|---|---|")
for q in ("take_another_step", "risky", "complete"):
    rate = sum(g[q] for g in golds) / len(golds)
    for m in ORDER:
        rows = runs[m]
        sc = [r["answers"][q]["noul"] for r in rows]
        lb = [r["gold"][q] for r in rows]
        lo, hi = auroc_ci(sc, lb)
        print("| %s (%.0f%%) | %s | %.0f%% | %.2f [%.2f, %.2f] |" % (q, 100 * rate, m, 100 * sum(s >= 0.5 for s in sc) / len(sc), auroc(sc, lb), lo, hi))

print("\n### Confidence on next_action: mean top probability when correct vs wrong\n")
for m in ORDER:
    rows = runs[m]
    top = lambda r: max(r["answers"]["next_action"]["probabilities"].values())
    c = [top(r) for r in rows if pred(r["answers"], "next_action") == r["gold"]["next_action"]]
    w = [top(r) for r in rows if pred(r["answers"], "next_action") != r["gold"]["next_action"]]
    print("- %s: correct %.2f (n=%d), wrong %.2f (n=%d)" % (m, sum(c) / max(1, len(c)), len(c), sum(w) / max(1, len(w)), len(w)))

print("\n### CPU latency per transcript (4 questions), p50 ms\n")
for m in ORDER:
    ms = sorted(r["ms"] for r in runs[m])
    print("- %s: %.0f ms" % (m, ms[len(ms) // 2]))
