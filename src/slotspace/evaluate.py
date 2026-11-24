"""Official STaRK means alongside top-100 per-query statistics."""

import numpy as np

METRICS = ("hit@1", "hit@5", "recall@20", "recall@100", "mrr", "rprecision")


def per_query_metrics(ranked: list[int], answers: set[int]) -> dict[str, float]:
    if not answers or len(ranked) != len(set(ranked)):
        raise ValueError("Metrics require nonempty answers and distinct ranked IDs")
    first = next((i for i, d in enumerate(ranked) if d in answers), None)
    return {
        "hit@1": float(first is not None and first < 1),
        "hit@5": float(first is not None and first < 5),
        "recall@20": len(set(ranked[:20]) & answers) / len(answers),
        "recall@100": len(set(ranked[:100]) & answers) / len(answers),
        "mrr": 0.0 if first is None else 1.0 / (first + 1),
        "rprecision": len(set(ranked[:len(answers)]) & answers) / len(answers),
    }


class Evaluation:
    def __init__(self, candidate_ids: list[int]):
        from stark_qa.evaluator import Evaluator

        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)) or min(candidate_ids) < 0:
            raise ValueError("Evaluator requires distinct nonnegative candidate IDs")
        self.candidates = set(candidate_ids)
        self.official = Evaluator(candidate_ids)
        self.rows: list[dict] = []
        self.official_rows: list[dict] = []
        self.empty_rankings = 0

    def add(self, qid: int, ranked: list[int], answers: list[int]) -> None:
        import torch

        if (len(ranked) > 100 or len(answers) == 0 or len(ranked) != len(set(ranked))
                or not set(ranked).issubset(self.candidates)
                or not set(answers).issubset(self.candidates)):
            raise ValueError(f"Invalid result/answer IDs or result length for query {qid}")
        ours = per_query_metrics(ranked, set(answers))
        # Encode the returned order to resolve exact score ties consistently in
        # all torchmetrics versions, without altering the saved retrieval scores.
        pred = {int(doc): float(len(ranked) - i) for i, doc in enumerate(ranked)}
        if ranked:
            official = self.official(pred, torch.tensor(answers, dtype=torch.long), metrics=list(METRICS))
        else:
            # The official API cannot express an empty ranking (a placeholder
            # incorrectly changes hit/rprecision). Empty retrieval has zero quality.
            official = {m: 0.0 for m in METRICS}
            self.empty_rankings += 1
        self.rows.append({"qid": qid, **ours})
        self.official_rows.append(official)

    def summary(self) -> dict:
        if not self.rows:
            raise ValueError("Cannot summarize an empty evaluation")
        ours = {m: 100 * float(np.mean([r[m] for r in self.rows])) for m in METRICS}
        official = {m: 100 * float(np.mean([r[m] for r in self.official_rows])) for m in METRICS}
        gaps = {m: abs(ours[m] - official[m]) for m in METRICS}
        for metric, delta in gaps.items():
            tolerance = 0.5 if metric == "mrr" else 0.1
            if delta > tolerance:
                raise ValueError(f"Official/per-query {metric} differ by {delta:.4f} points")
        return {"n_queries": len(self.rows), "official": official, "top100": ours,
                "agreement_points": gaps, "empty_rankings": self.empty_rankings,
                "empty_policy": "zero-quality extension; the upstream API has no empty-result representation"}
