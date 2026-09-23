"""Run every checkpoint over the agent-judge scenarios. Writes results/raw.json.

Modes: 'default' keeps the start of a long state (Laya's default), 'keep_end' keeps the end
(where an agent transcript's latest step lives).
"""
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
LAYA = os.path.join(HERE, "..", "models", "laya")
sys.path.insert(0, LAYA)
sys.path.insert(0, HERE)

import torch  # noqa: E402
import rl_agent_api  # noqa: E402
import rl_common  # noqa: E402
from scenarios import QUESTIONS, S, gold, state  # noqa: E402

torch.set_num_threads(os.cpu_count())
_orig = rl_common.build_sequence


def run(checkpoint, mode):
    keep_end = mode == "keep_end"
    rl_agent_api.build_sequence = lambda *a, **k: _orig(*a, truncate_left=keep_end, **k)
    agent = rl_agent_api.RLAgent(os.path.join(LAYA, checkpoint), device="cpu")
    rows = []
    for s in S:
        t = time.perf_counter()
        out = agent.system_one(state(s), QUESTIONS)
        ms = (time.perf_counter() - t) * 1000
        rows.append({"id": s["id"], "cat": s["cat"], "gold": gold(s), "answers": out["answers"],
                     "tokens": out["usage"]["input_tokens"], "ms": round(ms, 1)})
    return rows


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    res = {}
    for ck, name in [("", "english"), ("multilingual", "multilingual"), ("typed-decisions", "typed-decisions")]:
        for mode in ["default", "keep_end"]:
            key = "%s/%s" % (name, mode)
            t = time.time()
            res[key] = run(ck, mode)
            print(key, "done in %.0fs" % (time.time() - t), flush=True)
    with open(os.path.join(HERE, "results", "raw.json"), "w") as f:
        json.dump({"scenarios": len(S), "runs": res}, f, indent=1)
