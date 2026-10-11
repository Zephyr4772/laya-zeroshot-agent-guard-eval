"""Julia-1 vs Laya on the agent step-guard bench, identical plain-text input (scenarios.state_text) and questions.

Run twice:
  venvs/julia/Scripts/python.exe run_compare.py julia   -> results/compare_julia.json
  py -3.12 run_compare.py laya                           -> results/compare_laya.json  (all 3 checkpoints)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scenarios import QUESTIONS, S, gold, state_text  # noqa: E402

MODELS = os.path.join(HERE, "..", "models")


def run(predict, name):
    rows = []
    for s in S:
        t = time.perf_counter()
        a = predict(state_text(s), QUESTIONS)
        rows.append({"id": s["id"], "cat": s["cat"], "gold": gold(s), "answers": a,
                     "ms": round((time.perf_counter() - t) * 1000, 1)})
    print(name, "done", len(rows), "scenarios", flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1]
    out = {}
    if which == "julia":
        from julia import load_model
        e = load_model(os.path.join(MODELS, "Julia-1"), device="cpu", strict_encoding=True, max_length=8192, head_length=512)
        out["julia-1"] = run(lambda st, q: e.predict(state=st, questions=q)["answers"], "julia-1")
    else:
        os.environ.setdefault("USE_TF", "0")
        sys.path.insert(0, os.path.join(MODELS, "laya"))
        from rl_agent_api import RLAgent
        for sub, name in [("", "laya-english"), ("multilingual", "laya-multilingual"), ("typed-decisions", "laya-typed-decisions")]:
            ag = RLAgent(os.path.join(MODELS, "laya", sub), device="cpu")
            out[name] = run(lambda st, q: ag.system_one(st, q)["answers"], name)
    json.dump(out, open(os.path.join(HERE, "results", "compare_%s.json" % which), "w"), indent=1)
