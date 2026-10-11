### Accuracy by question (%)

| system | next_action | take_another_step | risky | complete | mean |
|---|---|---|---|---|---|
| always the most common answer | 44.3 | 55.7 | 85.2 | 85.2 | 67.6 |
| julia-1 | 23.0 | 47.5 | 85.2 | 77.0 | 58.2 |
| laya-typed-decisions | 32.8 | 32.8 | 85.2 | 65.6 | 54.1 |
| laya-multilingual | 24.6 | 42.6 | 85.2 | 29.5 | 45.5 |
| laya-english | 18.0 | 44.3 | 83.6 | 52.5 | 49.6 |

### next_action accuracy by category (%)

| system | done_clean | in_progress | loop | false_success | premature_claim | risky | safe_lookalike | blocked | dead_end | long |
|---|---|---|---|---|---|---|---|---|---|---|
| julia-1 | 71 | 0 | 0 | 25 | 33 | 0 | 60 | 0 | 33 | 25 |
| laya-typed-decisions | 71 | 29 | 14 | 0 | 0 | 0 | 80 | 100 | 33 | 25 |
| laya-multilingual | 43 | 14 | 43 | 0 | 0 | 0 | 40 | 50 | 67 | 25 |
| laya-english | 86 | 14 | 0 | 0 | 0 | 0 | 40 | 0 | 33 | 25 |

### What each model answers for next_action (gold: {'finish': 8, 'continue': 27, 'ask_human': 20, 'stop': 6})

- julia-1: {'finish': 37, 'continue': 13, 'stop': 11}
- laya-typed-decisions: {'finish': 25, 'continue': 20, 'ask_human': 14, 'stop': 2}
- laya-multilingual: {'finish': 25, 'stop': 24, 'continue': 8, 'ask_human': 4}
- laya-english: {'finish': 43, 'stop': 7, 'continue': 11}

### Yes/no questions: how often 'yes', and AUROC with 95% CI

| question (true rate) | system | says yes | AUROC [95% CI] |
|---|---|---|---|
| take_another_step (44%) | julia-1 | 48% | 0.46 [0.31, 0.61] |
| take_another_step (44%) | laya-typed-decisions | 56% | 0.35 [0.21, 0.49] |
| take_another_step (44%) | laya-multilingual | 95% | 0.45 [0.30, 0.59] |
| take_another_step (44%) | laya-english | 84% | 0.39 [0.25, 0.54] |
| risky (15%) | julia-1 | 0% | 0.49 [0.22, 0.76] |
| risky (15%) | laya-typed-decisions | 3% | 0.75 [0.57, 0.90] |
| risky (15%) | laya-multilingual | 3% | 0.63 [0.41, 0.84] |
| risky (15%) | laya-english | 2% | 0.68 [0.51, 0.84] |
| complete (15%) | julia-1 | 21% | 0.67 [0.48, 0.85] |
| complete (15%) | laya-typed-decisions | 33% | 0.71 [0.56, 0.84] |
| complete (15%) | laya-multilingual | 85% | 0.56 [0.40, 0.72] |
| complete (15%) | laya-english | 49% | 0.67 [0.47, 0.85] |

### Confidence on next_action: mean top probability when correct vs wrong

- julia-1: correct 0.78 (n=14), wrong 0.81 (n=47)
- laya-typed-decisions: correct 0.36 (n=20), wrong 0.35 (n=41)
- laya-multilingual: correct 0.52 (n=15), wrong 0.59 (n=46)
- laya-english: correct 0.47 (n=11), wrong 0.51 (n=50)

### CPU latency per transcript (4 questions), p50 ms

- julia-1: 246 ms
- laya-typed-decisions: 1151 ms
- laya-multilingual: 432 ms
- laya-english: 1078 ms
