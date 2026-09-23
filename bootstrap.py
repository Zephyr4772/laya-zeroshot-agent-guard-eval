"""Bootstrap 95% CIs for noul AUROCs (resampling scenarios)."""
import json, os, random
from analyze import auroc, prob, runs
random.seed(0)
for q in ["take_another_step", "risky", "complete"]:
    for k in ["english/default", "multilingual/default", "typed-decisions/default"]:
        rows = runs[k]
        pt = auroc([prob(r["answers"], q) for r in rows], [r["gold"][q] for r in rows])
        bs = []
        for _ in range(2000):
            s = [random.choice(rows) for _ in rows]
            v = auroc([prob(r["answers"], q) for r in s], [r["gold"][q] for r in s])
            if v == v: bs.append(v)
        bs.sort()
        print("%-18s %-24s AUROC %.2f  95%% CI [%.2f, %.2f]" % (q, k, pt, bs[int(.025*len(bs))], bs[int(.975*len(bs))]))
