"""Rebuild M8 tables, paired statistics and SVG figures from saved run artifacts only."""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import read_jsonl, sha256, write_json
from .evaluate import METRICS
from .protocol import catalog
from .stats import bootstrap_ci, discordant_rate, holm, mde_points, paired_t, randomization_test


def generate_pilot_report(run: Path, out: Path):
    """Report saved preliminary validation measurements without invoking any model."""
    info = json.loads((run / "pilot.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    if info["purpose"] != "pilot" or info["formal_acceptance"] or meta["status"] != "ok":
        raise ValueError("Pilot reporting requires a completed, explicitly non-production run")
    quality = pd.read_parquet(run / "per_query.parquet")
    cost = pd.read_parquet(run / "counters.parquet")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    methods = list(metrics)
    rows, comparisons, pvals = [], {}, {}
    reference = quality[quality.method == "hybrid"].set_index("qid")
    if reference.empty or reference.index.duplicated().any():
        raise ValueError("Pilot report needs distinct hybrid reference query IDs")
    lines = ["# Preliminary validation pilot - NOT a formal M8 result", "",
             "Answer-closed diagnostic corpus; full KB dictionary; zero-shot scorer at T=1; raw lexicon counts.",
             "Validation is reused for tuning and measurement. No confirmatory significance or conformal claim is made.",
             f"Stage: {info['stage']}; documents: {info['budget']['documents']}; "
             f"measured repeats: {info['budget']['repeats']}.", "",
             "| Method | n | Hit@1 | Recall@20 | MRR | Recall@100 | p50 client ms | Mean candidates | Mean facts |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = info.get("data_summary")
    if summary:
        lines[5:5] = [f"Calibration: {len(summary['calibration_qids'])} even-train queries; "
                      f"validation: {len(summary['validation_qids'])} queries; text facts: "
                      f"{summary['text_facts']}/{summary['source_text_facts_in_selected_documents']} retained.", ""]
    for method in methods:
        q = quality[quality.method == method].set_index("qid")
        c = cost[cost.method == method].set_index("qid")
        if (q.index.duplicated().any() or c.index.duplicated().any() or set(q.index) != set(reference.index)
                or set(c.index) != set(q.index) or len(q) != metrics[method]["n_queries"]
                or len(q) != info["budget"]["validation"] or not np.isfinite(c.latency_ms).all()
                or not np.isfinite(q[list(METRICS)].to_numpy()).all()):
            raise ValueError("Pilot report cannot pair missing/duplicate query IDs or invalid timings")
        means = {k: 100 * float(q[k].mean()) for k in METRICS}
        if any(abs(means[k] - metrics[method]["top100"][k]) > 1e-6 for k in METRICS):
            raise ValueError("Pilot means differ from their saved per-query vectors")
        latency = float(c.latency_ms.median())
        mean_facts = float(c.facts_scored.mean())
        lines.append(f"| {method} | {len(q)} | {means['hit@1']:.2f} | {means['recall@20']:.2f} | "
                     f"{means['mrr']:.2f} | {means['recall@100']:.2f} | {latency:.2f} | "
                     f"{c.candidates.mean():.1f} | {mean_facts:.1f} |")
        rows.append({"method": method, **means, "median_latency_ms": latency, "mean_facts_scored": mean_facts})
        if method != "hybrid":
            q = q.loc[reference.index]
            a, b = q["mrr"].to_numpy(), reference["mrr"].to_numpy()
            pvals[method] = randomization_test(a, b)
            comparisons[method] = {"mrr_difference_ci95": list(100 * np.asarray(bootstrap_ci(a, b))),
                                   "randomization_p": pvals[method],
                                   "discordant_hit1": discordant_rate(q["hit@1"], reference["hit@1"])}
    adjusted = holm(pvals)
    for method in comparisons:
        comparisons[method]["holm_p"] = adjusted[method]
        comparisons[method]["inference"] = "descriptive only; validation selection bias and small conditional corpus"
    selection = info["selection"]
    lines += ["", "## M7 replay/tuning", "",
              "Selected settings (pilot only): `" + json.dumps({
                  "planner": selection["config"]["planner"], "scoring": selection["config"]["scoring"]}) + "`.",
              f"Plan regret: {selection['metrics']['plan_regret']}; observed labelled-segment miss rate: "
              f"{selection['metrics']['conformal_miss_rate']}.",
              "Full-corpus calibration/validation are incomplete; production freezing and H5 approval are forbidden.", ""]
    if info["budget"]["documents"] <= 100:
        lines.append("Recall@100 is a ceiling in this small answer-closed corpus, not a coverage acceptance result.")
    lines += ["", "No main-model training, PACC rate fitting, official test/human evaluation, GPU or Vespa measurement "
              "is implied. More queries/candidates and independent held-out evaluation are required.", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    result = {"purpose": "pilot", "formal_acceptance": False, "rows": rows, "comparisons": comparisons,
              "n_queries": len(reference), "mde_assumed_discordance_0_2_points": mde_points(len(reference), 0.2),
              "source_run": str(run.resolve())}
    write_json(out.with_suffix(".json"), result)
    return result


def load_runs(plan_path: Path, execution: Path | None, include_fixtures=False):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    state = json.loads((execution / "execution.json").read_text(encoding="utf-8")) if execution else {"jobs": {}, "references": {}}
    if execution and state["plan_sha256"] != sha256(plan_path):
        raise ValueError("Execution log belongs to another protocol")
    runs = {}
    for job in plan["jobs"]:
        recorded = state["jobs"].get(job["id"])
        if recorded is None or recorded["status"] != "ok":
            continue
        path = Path(recorded["path"])
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        cfg = meta["config"]
        target = job.get("alias_of", job["id"])
        if (meta["status"] != "ok" or cfg["protocol_job"] != target or cfg["dataset"] != job["dataset"]
                or cfg["plan_sha256"] != sha256(plan_path)
                or cfg["frozen_sha256"] != plan["frozen_sha256"]
                or cfg["queries_sha256"] != plan["datasets"][job["dataset"]]["sha256"]):
            raise ValueError("Run metadata conflicts with the approved protocol")
        if meta["measurement_source"] != "runtime" and not include_fixtures:
            continue
        quality = pd.read_parquet(path / "per_query.parquet")
        cost = pd.read_parquet(path / "counters.parquet")
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        if (not len(quality) or quality.qid.duplicated().any() or cost.qid.duplicated().any()
                or set(quality.qid) != set(cost.qid) or len(quality) != metrics["n_queries"]
                or not set(METRICS) <= set(quality)):
            raise ValueError("Run vectors/counters are missing, duplicated or misaligned")
        values = quality[list(METRICS)].to_numpy()
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Nonfinite per-query metric values")
        for metric in METRICS:
            if not np.isclose(100 * quality[metric].mean(), metrics["top100"][metric], atol=1e-6):
                raise ValueError("Saved means do not reproduce the saved per-query vectors")
        for counter in ("bm25_postings", "dense_searches", "value_postings", "candidates", "facts_scored"):
            if counter not in cost or not np.isfinite(cost[counter]).all() or (cost[counter] < 0).any():
                raise ValueError(f"Invalid work counter: {counter}")
        latency = cost.latency_ms.dropna()
        if not np.isfinite(latency).all() or (latency < 0).any():
            raise ValueError("Invalid measured latency")
        runs[(job["experiment"]["id"], job["dataset"])] = {
            "path": path, "quality": quality.set_index("qid"), "cost": cost, "metrics": metrics,
            "job": job, "source": meta["measurement_source"]}
    references = {}
    for name, ref in state.get("references", {}).items():
        path = Path(ref["path"])
        if sha256(path) != ref["sha256"]:
            raise ValueError("Copied report diagnostic changed")
        references[name] = path
    return plan, runs, references


def comparisons(runs: dict, dataset: str, experiments: list[str]):
    reference = runs.get(("A0", dataset))
    family = [name for name in experiments if name != "A0"]
    results = {}
    pvalues = {name: 1.0 for name in family}
    if reference is not None:
        for name in family:
            other = runs.get((name, dataset))
            if other is None:
                continue
            a, b = reference["quality"], other["quality"]
            if set(a.index) != set(b.index):
                raise ValueError("Paired comparisons require exactly the same query IDs; no inner-join censoring")
            ids = sorted(a.index, key=str)
            metrics = {}
            for metric in ("hit@1", "mrr", "recall@100", "role_confusion@10", "plan_regret"):
                if metric not in a or metric not in b:
                    continue
                chosen = ids
                if metric == "plan_regret":
                    if set(a.index[a[metric].notna()]) != set(b.index[b[metric].notna()]):
                        raise ValueError("Planner-regret label masks differ across runs")
                    chosen = [qid for qid in ids if pd.notna(a.at[qid, metric])]
                    if not chosen:
                        continue
                av, bv = a.loc[chosen, metric].to_numpy(float), b.loc[chosen, metric].to_numpy(float)
                if not np.isfinite(av).all() or not np.isfinite(bv).all():
                    raise ValueError("Nonfinite paired-comparison vectors")
                difference = av - bv
                pt = (1.0 if np.allclose(difference, 0) else
                      0.0 if np.allclose(difference, difference[0]) else paired_t(av, bv)) if len(chosen) >= 2 else None
                random = randomization_test(av, bv)
                mean, low, high = bootstrap_ci(av, bv)
                metrics[metric] = {"n": len(chosen), "difference_points": 100 * mean, "ci95_points": [100 * low, 100 * high],
                                   "p_t": pt, "p_randomization": random}
            discordance = discordant_rate(a.loc[ids, "hit@1"], b.loc[ids, "hit@1"])
            results[name] = {"n": len(ids), "discordance": discordance,
                             "mde_points": mde_points(len(ids), discordance, m=max(1, len(family))),
                             "metrics": metrics}
            pvalues[name] = metrics["hit@1"]["p_randomization"]
    for metric in ("hit@1", "mrr", "recall@100", "role_confusion@10", "plan_regret"):
        for field in ("p_t", "p_randomization"):
            family_p = {name: (results.get(name, {}).get("metrics", {}).get(metric, {}).get(field)
                               if results.get(name, {}).get("metrics", {}).get(metric, {}).get(field) is not None else 1.0)
                        for name in family}
            adjusted_metric = holm(family_p)
            for name in results:
                if metric in results[name]["metrics"]:
                    results[name]["metrics"][metric][field + "_holm"] = adjusted_metric[name]
    adjusted = holm(pvalues)
    for name, value in results.items():
        value["p_holm"] = adjusted[name]
    return results


def _value(runs, name, dataset, key):
    run = runs.get((name, dataset))
    if run is None:
        return None
    if key in run["quality"]:
        return 100 * float(run["quality"][key].mean())
    return run["metrics"].get(key)


def pass_lines(runs, paired, references):
    result = {}

    def measured(keys):
        return all(_value(runs, *key) is not None for key in keys)

    def margin(other):
        stats = paired.get(other)
        if stats is None:
            return "NOT RUN"
        delta = stats["metrics"]["hit@1"]["difference_points"]
        return ("PASS" if delta > 0 and delta >= stats["mde_points"] and stats["p_holm"] < 0.05
                else "FAIL" if delta <= -max(stats["mde_points"], 1e-12) and stats["p_holm"] < 0.05
                else "NOT DETECTED")

    result["A1"] = margin("A1")
    for name in ("A2", "A3"):
        keys = [("A0", "role_swap_test", "role_confusion@10"), (name, "role_swap_test", "role_confusion@10")]
        lower = (_value(runs, *keys[0]) < _value(runs, *keys[1])) if measured(keys) else None
        if lower is None:
            result[name] = "NOT RUN"
        elif name == "A2":
            result[name] = "PASS" if lower else "FAIL"
        else:
            result[name] = "PASS" if lower and margin(name) == "PASS" else (
                "FAIL" if margin(name) == "FAIL" else "NOT DETECTED")
    def regret(name):
        row = runs.get((name, "val"))
        return row["metrics"].get("planner", {}).get("regret") if row else None
    for name in ("A4", "A5", "A13"):
        a, b = regret("A0"), regret(name)
        if a is None or b is None or (name == "A4" and name not in paired):
            result[name] = "NOT RUN"
        elif name == "A4":
            result[name] = "PASS" if a < b and margin(name) == "PASS" else "NOT DETECTED"
        else:
            result[name] = "PASS" if a < b else "FAIL"
    keys = [("A0", "prime_test", "recall@100"), ("A6", "prime_test", "recall@100")]
    result["A6"] = ("PASS" if _value(runs, *keys[0]) >= _value(runs, *keys[1]) + 1 else "FAIL") if measured(keys) else "NOT RUN"
    keys = [("A0", "role_swap_test", "hit@5"), ("A7", "role_swap_test", "hit@5"),
            ("A0", "prime_test", "recall@100"), ("A7", "prime_test", "recall@100")]
    result["A7"] = ("PASS" if _value(runs, *keys[0]) >= _value(runs, *keys[1]) + 3
                     and _value(runs, *keys[2]) >= _value(runs, *keys[3]) - 1 else "FAIL") if measured(keys) else "NOT RUN"
    diagnostic = json.loads(references["pacc"].read_text()) if "pacc" in references else None
    result["A8"] = ("PASS" if diagnostic["pacc_relative_error"] <= 0.05
                    and diagnostic["pacc_relative_error"] < diagnostic["raw_relative_error"] else "FAIL"
                    ) if diagnostic and diagnostic["pacc_relative_error"] is not None else "NOT RUN"
    result["A9"] = "DESCRIPTIVE" if all((f"A9_{k}", "role_swap_test") in runs for k in ("p", "p_over_H", "ia")) else "NOT RUN"
    conformal = []
    for alpha in (0.05, 0.1, 0.2):
        row = runs.get((f"A10_{alpha}", "val"))
        if row is None:
            conformal = []
            break
        miss = row["metrics"].get("planner", {}).get("miss_rate")
        if miss is None:
            conformal = []
            break
        work = float((row["cost"].bm25_postings + row["cost"].value_postings).mean())
        conformal.append((miss <= alpha + 0.02, work))
    result["A10"] = ("PASS" if all(ok for ok, _ in conformal)
                      and all(a[1] >= b[1] for a, b in zip(conformal, conformal[1:])) else "FAIL") if len(conformal) == 3 else "NOT RUN"
    keys = [("A0", "role_swap_test", "hit@1"), ("A11", "role_swap_test", "hit@1")]
    storage = _value(runs, "A11", "role_swap_test", "additional_index_bytes_per_doc")
    result["A11"] = ("PASS" if _value(runs, *keys[0]) - _value(runs, *keys[1]) <= 1
                      and storage <= 2048 else "FAIL") if measured(keys) and storage is not None else "NOT RUN"
    keys = [("A0", "role_swap_test", "hit@1"), ("A12", "role_swap_test", "hit@1")]
    result["A12"] = ("PASS" if _value(runs, *keys[0]) >= _value(runs, *keys[1]) else "FAIL") if measured(keys) else "NOT RUN"
    row = runs.get(("A14", "val"))
    result["A14"] = "DESCRIPTIVE" if row and row["metrics"].get("planner", {}).get("heldout_slot_accuracy") is not None else "NOT RUN"
    return result


def _fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def scatter_svg(points, x_key, title):
    width, height = 720, 400
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="35" y="30" font-size="17">{html.escape(title)}</text>']
    if not points:
        output.append('<text x="120" y="190" font-size="20">NOT RUN - no completed measurements</text>')
    else:
        xmax = max(max(p[x_key] for p in points) * 1.1, 1)
        output += ['<path d="M70 55 V340 H680" fill="none" stroke="black"/>',
                   '<text x="10" y="65">100</text><text x="15" y="340">0</text>',
                   '<text x="65" y="360">0</text>',
                   f'<text x="620" y="360">{xmax:.3g}</text>',
                   f'<text x="150" y="385">{"Posting-list entries" if x_key == "work" else "Online latency (ms)"} (x), Recall@100 (%) (y)</text>']
        colors = {"A0": "#1764ab", "B2": "#dd7b0a", "B6": "#17905d"}
        for p in points:
            x, y = 70 + 590 * p[x_key] / xmax, 340 - 280 * p["recall"] / 100
            output.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{colors[p["id"]]}"/>')
            output.append(f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" font-size="12">{html.escape(p["id"])}</text>')
    return "\n".join([*output, "</svg>"])


def reliability_svg(samples, fixture=False):
    for row in samples:
        if row["correct"] not in (True, False, 0, 1):
            raise ValueError("Invalid reliability truth label")
        for field in ("confidence_before", "confidence_after"):
            value = row.get(field)
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Invalid reliability confidence")
    output = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="420" viewBox="0 0 800 420">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="25" y="25" font-size="16">{"SYNTHETIC FIXTURE - " if fixture else ""}Reliability - query validation / document calibration</text>']
    for j, style in enumerate(("query", "doc")):
        left = 60 + j * 390
        rows = [r for r in samples if r["style"] == style]
        output.append(f'<text x="{left}" y="55">{style}</text>')
        if not rows:
            output.append(f'<text x="{left}" y="190">NOT RUN - no data</text>')
            continue
        output.append(f'<path d="M{left} 80 V340 H{left + 280} M{left} 340 L{left + 280} 80" fill="none" stroke="gray"/>')
        output.append(f'<text x="{left - 12}" y="355">0</text><text x="{left + 275}" y="355">1</text>')
        for field, color in (("confidence_before", "#dd7b0a"), ("confidence_after", "#1764ab")):
            for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
                selected = [r for r in rows if r.get(field) is not None and (r[field] >= lo if lo == 0 else r[field] > lo)
                            and r[field] <= hi]
                if selected:
                    x = left + 280 * np.mean([r[field] for r in selected])
                    y = 340 - 260 * np.mean([r["correct"] for r in selected])
                    output.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    return "\n".join([*output, '<text x="35" y="380" font-size="12">Silver query labels / KB document calibration; these are not human gold labels.</text>',
                      '<text x="160" y="405">Orange: before temperature; blue: after temperature</text>', "</svg>"])


def generate_report(plan_path: Path, execution: Path | None, out: Path, *, include_fixtures=False):
    plan, runs, references = load_runs(plan_path, execution, include_fixtures)
    figures = out.parent / (out.stem + "_figures")
    data_path = out.with_suffix(".json")
    if out.exists() or figures.exists() or data_path.exists():
        raise FileExistsError("Report or its figures/data already exist; select a new output path")
    figures.mkdir(parents=True)
    baseline = ["A0", *[s.id for s in catalog() if s.id.startswith("B") and s.id != "B7"]]
    ablations = [s.id for s in catalog() if s.id.startswith("A")]
    paired = {d: comparisons(runs, d, baseline) for d in ("prime_test", "human", "role_swap_test")}
    ablation_stats = comparisons(runs, "role_swap_test", ablations)
    lines = ["# SlotSpace v0 report", "",
             "**DRAFT / NOT RUN**" if plan["draft"] or not runs else
             "**SYNTHETIC FIXTURE REPORT - NOT BENCHMARK EVIDENCE**" if include_fixtures else "**Completed frozen-run evidence**",
             "", f"Configuration SHA-256{' (not frozen)' if plan['draft'] else ' (frozen)'}: `{plan['frozen_sha256']}`.",
             "Missing runs are NOT RUN, never zero-filled. Human-written results are directional evidence only.",
             "Differences are A0 minus comparator. Holm uses the prespecified table family; unrun comparisons remain unreported.",
             "95% percentile bootstrap intervals use 1,000 resamples; two-sided randomization uses 10,000 permutations.", ""]
    for title, dataset in (("Table 1: Prime test", "prime_test"), ("Table 1b: human-written", "human"),
                           ("Table 2: role-swap test", "role_swap_test")):
        lines += [f"## {title}", "", "| System | n | Hit@1 | Hit@5 | R@20 | MRR | R@100 | RC@10 | Work | ms | Holm p | MDE |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in baseline:
            row = runs.get((name, dataset))
            if row is None:
                lines.append(f"| {name} (NOT RUN) | - | - | - | - | - | - | - | - | - | - | - |")
                continue
            q, c, comp = row["quality"], row["cost"], paired[dataset].get(name, {})
            values = [len(q), *[100 * q[m].mean() for m in ("hit@1", "hit@5", "recall@20", "mrr", "recall@100")],
                      100 * q["role_confusion@10"].mean() if "role_confusion@10" in q else None,
                      (c.bm25_postings + c.value_postings).mean(),
                      float(c.latency_ms.dropna().median()) if c.latency_ms.notna().any() else None,
                      comp.get("p_holm"), comp.get("mde_points")]
            lines.append("| " + name + " | " + " | ".join(_fmt(v) for v in values) + " |")
        lines.append("")
    outcomes = pass_lines(runs, ablation_stats, references)
    lines += ["## Table 3: ablations", "", "| Condition | n (role-swap) | Hit@1 | Hit@5 | R@100 | RC@10 | Holm p | MDE | Decision |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name in ablations:
        row = runs.get((name, "role_swap_test"))
        comp = ablation_stats.get(name, {})
        values = ([len(row["quality"]), *[100 * row["quality"][m].mean() for m in
                   ("hit@1", "hit@5", "recall@100", "role_confusion@10")], comp.get("p_holm"), comp.get("mde_points")]
                  if row is not None else [None] * 7)
        decision = outcomes.get(name.split("_")[0], "REFERENCE" if row is not None else "NOT RUN")
        lines.append("| " + name + " | " + " | ".join(_fmt(v) for v in values) + f" | {decision} |")
    planner_names = ["A0", "A4", "A5", "A12", "A13", "A14", "B7",
                     "A10_0.05", "A10_0.1", "A10_0.2", "A10_all"]
    planner_stats = comparisons(runs, "val", planner_names)
    lines += ["", "## Table 4: planner (validation only)", "",
              "MDE below is for paired Hit@1, not a claimed power bound for ECE or regret.", "",
              "| System | n queries | ECE before | ECE after | Slot acc | Miss rate | Mean regret | Regret 95% CI | Holm p (regret) | Hit@1 MDE | Held-out accuracy / n |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for name in planner_names:
        run = runs.get((name, "val"))
        p = run["metrics"].get("planner", {}) if run else {}
        ci = p.get("regret_ci95")
        values = [len(run["quality"]) if run else None, p.get("ece_before"), p.get("ece_after"), p.get("slot_accuracy"),
                  p.get("miss_rate"), p.get("regret")]
        lines.append("| " + name + " | " + " | ".join(_fmt(v, 4) for v in values)
                     + " | " + (str(ci) if ci else "-") + " | "
                     + _fmt(planner_stats.get(name, {}).get("metrics", {}).get("plan_regret", {}).get("p_randomization_holm"), 4)
                     + " | " + _fmt(planner_stats.get(name, {}).get("mde_points")) + " | "
                     + (f"{_fmt(p.get('heldout_slot_accuracy'), 4)} / {p['heldout_slot_n']}" if p.get("heldout_slot_n") else "-") + " |")
    points = []
    for name in ("A0", "B2", "B6"):
        row = runs.get((name, "prime_test"))
        if row:
            cost = row["cost"]
            points.append({"id": name, "work": float((cost.bm25_postings + cost.value_postings).mean()),
                           "latency": float(cost.latency_ms.dropna().median()) if cost.latency_ms.notna().any() else None,
                           "recall": float(100 * row["quality"]["recall@100"].mean())})
    prefix = "SYNTHETIC FIXTURE - " if include_fixtures else ""
    (figures / "work.svg").write_text(scatter_svg(points, "work", prefix + "Recall@100 against posting work"), encoding="utf-8")
    (figures / "latency.svg").write_text(scatter_svg([p for p in points if p["latency"] is not None],
                                                    "latency", prefix + "Recall@100 against measured online latency"), encoding="utf-8")
    samples = []
    if "calibration_points" in references:
        samples += [r for r in pd.read_parquet(references["calibration_points"]).to_dict(orient="records") if r["style"] == "doc"]
    query_run = runs.get(("A0", "val"))
    if query_run and (query_run["path"] / "planner_diagnostics.jsonl").exists():
        for d in read_jsonl(query_run["path"] / "planner_diagnostics.jsonl"):
            if len(d["confidence_before"]) != len(d["confidence"]) or len(d["confidence"]) != len(d["correct"]):
                raise ValueError("Misaligned planner reliability vectors")
            samples.extend({"style": "query", "confidence_before": a, "confidence_after": b, "correct": c}
                           for a, b, c in zip(d["confidence_before"], d["confidence"], d["correct"]))
    (figures / "reliability.svg").write_text(reliability_svg(samples, include_fixtures), encoding="utf-8")
    lines += ["", "## Figures", "", f"![Figure 1a: work]({figures.name}/work.svg)",
              f"![Figure 1b: latency]({figures.name}/latency.svg)",
              f"![Figure 2: reliability]({figures.name}/reliability.svg)", "",
              "## Negative and inconclusive results", ""]
    lines += [f"- {name}: **{status}**." for name, status in outcomes.items() if status != "PASS"]
    lines += ["", "No missing GPU job, human checkpoint, baseline or ablation is treated as a successful experiment.",
              "Paired statistical details, intervals, discordance and per-comparison MDE are in the adjacent JSON file.", ""]
    summary = {"draft": plan["draft"], "completed_rows": len(runs), "pass_lines": outcomes,
               "paired_baselines": paired, "paired_ablations": ablation_stats, "paired_planner": planner_stats,
               "plan_sha256": sha256(plan_path), "fixtures_included": include_fixtures}
    with out.open("x", encoding="utf-8") as dest:
        dest.write("\n".join(lines))
    with data_path.open("x", encoding="utf-8") as dest:
        json.dump(summary, dest, indent=2, allow_nan=False)
    return summary
