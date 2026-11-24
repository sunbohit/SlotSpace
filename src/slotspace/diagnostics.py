"""Held-out quantification diagnostics; never evaluate PACC on its own fitted rates."""

import numpy as np
import pandas as pd

from .reference import pacc_binary


def pacc_diagnostic(rows: pd.DataFrame) -> dict:
    required = {"doc_id", "slot", "pi", "native"}
    if not required <= set(rows):
        raise ValueError("Invalid distant-supervision diagnostic rows")
    if rows.empty:
        return {"protocol": "rates on even doc_id; quantification error on odd doc_id", "slots": {},
                "true_total": 0, "raw_relative_error": None, "pacc_relative_error": None}
    if (not np.isfinite(rows.pi).all() or not rows.pi.between(0, 1).all()
            or rows.native.dtype != bool):
        raise ValueError("Invalid distant-supervision probabilities or binary truth labels")
    results = {}
    raw_error = corrected_error = true_total = 0.0
    for slot, group in rows.groupby("slot", sort=True):
        calibration = group[group.doc_id % 2 == 0]
        holdout = group[group.doc_id % 2 == 1]
        positives, negatives = calibration[calibration.native], calibration[~calibration.native]
        tpr = float(positives.pi.mean()) if len(positives) else None
        fpr = float(negatives.pi.mean()) if len(negatives) else None
        stable = tpr is not None and fpr is not None and tpr - fpr >= 0.05
        if not len(holdout):
            results[slot] = {"status": "NOT RUN", "reason": "no odd-document holdout pairs"}
            continue
        truth, raw = int(holdout.native.sum()), float(holdout.pi.sum())
        corrected = len(holdout) * pacc_binary(float(holdout.pi.mean()), tpr, fpr) if stable else raw
        raw_error += abs(raw - truth)
        corrected_error += abs(corrected - truth)
        true_total += truth
        results[slot] = {"n_calibration": len(calibration), "n_holdout": len(holdout), "truth": truth,
                         "raw": raw, "corrected": corrected, "tpr": tpr, "fpr": fpr,
                         "status": "corrected" if stable else "raw fallback"}
    return {"protocol": "rates on even doc_id; quantification error on odd doc_id",
            "slots": results, "true_total": int(true_total),
            "raw_relative_error": raw_error / true_total if true_total else None,
            "pacc_relative_error": corrected_error / true_total if true_total else None}
