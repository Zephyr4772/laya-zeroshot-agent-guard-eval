"""Long-transcript probe: pad the G* scenarios until they exceed the state budget, compare default vs keep_end."""
import copy
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run  # noqa: E402  (sets up paths + patching)
from scenarios import QUESTIONS, S, gold, state  # noqa: E402

PAD = [("agent", "Step %d: reading src/module_%02d.ts, nothing relevant to the task here" % (i, i)) for i in range(60)]
longs = []
for s in S:
    if s["cat"] == "long":
        x = copy.deepcopy(s)
        x["steps"] = x["steps"][:1] + PAD + x["steps"][1:]
        longs.append(x)

out = {}
for ck, name in [("", "english"), ("multilingual", "multilingual"), ("typed-decisions", "typed-decisions")]:
    for mode in ["default", "keep_end"]:
        keep_end = mode == "keep_end"
        run.rl_agent_api.build_sequence = lambda *a, _k=keep_end, **k: run._orig(*a, truncate_left=_k, **k)
        agent = run.rl_agent_api.RLAgent(os.path.join(run.LAYA, ck), device="cpu")
        rows = []
        for s in longs:
            a = agent.system_one(state(s), QUESTIONS)["answers"]
            rows.append((s["id"], gold(s)["next_action"], a["next_action"]["choice"], gold(s)["risky"], a["risky"]["noul"],
                         agent.tok(json.dumps(state(s)))["input_ids"].__len__()))
        out["%s/%s" % (name, mode)] = rows
        print("%-26s" % ("%s/%s" % (name, mode)),
              "  ".join("%s gold=%s pred=%s risky_p=%.2f(gold %s) state_tokens=%d" % r for r in rows), flush=True)
json.dump(out, open(os.path.join(HERE, "results", "truncation.json"), "w"), indent=1)
