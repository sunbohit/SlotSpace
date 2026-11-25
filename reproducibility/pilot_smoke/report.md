# Preliminary validation pilot - NOT a formal M8 result

Answer-closed diagnostic corpus; full KB dictionary; zero-shot scorer at T=1; raw lexicon counts.
Validation is reused for tuning and measurement. No confirmatory significance or conformal claim is made.
Stage: smoke; documents: 64; measured repeats: 1.
Calibration: 4 even-train queries; validation: 4 queries; text facts: 52/450 retained.


| Method | n | Hit@1 | Recall@20 | MRR | Recall@100 | p50 client ms | Mean candidates | Mean facts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 4 | 50.00 | 100.00 | 64.77 | 100.00 | 0.24 | 32.0 | 0.0 |
| dense | 4 | 25.00 | 100.00 | 41.37 | 100.00 | 260.57 | 32.0 | 0.0 |
| hybrid | 4 | 50.00 | 100.00 | 75.00 | 100.00 | 251.37 | 47.2 | 0.0 |
| bound | 4 | 100.00 | 100.00 | 100.00 | 100.00 | 3151.54 | 47.5 | 14.5 |

## M7 replay/tuning

Selected settings (pilot only): `{"planner": {"beta": 0.5, "eps": 1.0, "alpha": 0.2, "r_max": 6, "weight_objective": "p", "window_words": 8, "use_pacc": false}, "scoring": {"mode": "bound", "combine": "max", "a": 1.0, "b": 1.0, "precision": "float", "lambda_B": 1.0, "lambda_D": 1.0, "lambda_L": 0.0}}`.
Plan regret: 0.0; observed labelled-segment miss rate: 0.0.
Full-corpus calibration/validation are incomplete; production freezing and H5 approval are forbidden.

Recall@100 is a ceiling in this small answer-closed corpus, not a coverage acceptance result.

No main-model training, PACC rate fitting, official test/human evaluation, GPU or Vespa measurement is implied. More queries/candidates and independent held-out evaluation are required.
