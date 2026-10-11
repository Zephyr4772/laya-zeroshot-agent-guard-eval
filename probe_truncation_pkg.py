"""Follow-up to probe_truncation.py, prompted by discussion #15 on convaiinnovations/laya.

Same padded long transcripts (G01-G04, ~1,650 state tokens), run through the `laya` package (`laya.load`), whose
Agent truncates list states from the left (keeps the newest turn) and str/dict states from the right.
Input shapes:
  A dict  {task, steps}           -- what the original study passed (control; keeps the head)
  B list  [task, step1, ..., stepN] -- conversation list (keeps the tail; the task may be cut)
  C dict  {task, last 8 steps}    -- "send only the last few steps"
Writes results/truncation_pkg.json.
"""
import copy
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import laya  # noqa: E402
from scenarios import QUESTIONS, S, gold  # noqa: E402

LAYA = os.path.join(HERE, "..", "models", "laya")
PAD = [("agent", "Step %d: reading src/module_%02d.ts, nothing relevant to the task here" % (i, i)) for i in range(60)]
LONG = []
for s in S:
    if s["cat"] == "long":
        x = copy.deepcopy(s)
        x["steps"] = x["steps"][:1] + PAD + x["steps"][1:]
        LONG.append(x)

turns = lambda s: ["%s: %s" % (r, t) for r, t in s["steps"]]
SHAPES = {
    "A_dict_full": lambda s: {"task": s["task"], "steps": turns(s)},
    "B_list_turns": lambda s: ["task: " + s["task"]] + turns(s),
    "C_dict_last8": lambda s: {"task": s["task"], "steps": turns(s)[-8:]},
}

out = {"laya_version": laya.__version__, "rows": []}
for sub, name in [(None, "english"), ("multilingual", "multilingual"), ("typed-decisions", "typed-decisions")]:
    agent = laya.load(LAYA, subfolder=sub, device="cpu")
    for shape, mk in SHAPES.items():
        for s in LONG:
            a = agent.predict(mk(s), QUESTIONS)["answers"]
            row = {"checkpoint": name, "shape": shape, "id": s["id"], "gold_next_action": gold(s)["next_action"],
                   "pred_next_action": a["next_action"]["choice"], "gold_risky": gold(s)["risky"],
                   "risky_p": round(a["risky"]["noul"], 4)}
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
json.dump(out, open(os.path.join(HERE, "results", "truncation_pkg.json"), "w"), indent=1)
