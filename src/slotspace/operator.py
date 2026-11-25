"""Explicit human actions for an agent-free VM, not automatic experiment approvals."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .artifacts import sha256, write_json
from .protocol import authorize
from .reproduce import ReviewRequired, review_evidence


def approve_review(work: Path, gate: str, schema: Path, out: Path, *, confirmed: bool = False) -> dict:
    if not confirmed:
        raise ValueError("Approval requires --yes after the operator has reviewed the labelled samples")
    evidence = review_evidence(gate, work, schema)
    entry = {"approved": True, "evidence_sha256": evidence["evidence_sha256"]}
    records = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if not isinstance(records, dict) or set(records) - {"H1", "H2", "H3", "H4"}:
        raise ValueError("Review approvals must be a JSON object keyed by H1-H4")
    if gate in records and records[gate] != entry:
        raise FileExistsError("Existing approval has different evidence; revoke it explicitly before approving again")
    records[gate] = entry
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".review-", dir=out.parent) as temp:
        candidate = Path(temp) / "reviews.json"
        write_json(candidate, records)
        candidate.replace(out)
    return entry


def approve_h5(plan: Path, out: Path, *, confirmed: bool = False) -> dict:
    if not confirmed:
        raise ValueError("H5 approval requires --yes after reviewing the frozen configuration and final plan")
    if out.exists():
        raise FileExistsError("Do not overwrite an existing H5 authorization")
    proposal = json.loads(plan.read_text(encoding="utf-8"))
    if proposal["draft"]:
        raise ValueError("A draft cannot receive H5 approval")
    entry = {"approved": True, "frozen_sha256": proposal["frozen_sha256"], "plan_sha256": sha256(plan)}
    with TemporaryDirectory(prefix="slotspace-h5-") as temp:
        candidate = Path(temp) / "approval.json"
        write_json(candidate, entry)
        authorize(plan, candidate)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as target:
        target.write(json.dumps(entry, indent=2, allow_nan=False) + "\n")
    return entry


def status(work: Path) -> dict:
    path = work / "reproduction.json"
    if not path.is_file():
        return {"status": "NOT STARTED", "next": "Run slotspace.reproduce --dry-run, then start a new --work"}
    state = json.loads(path.read_text(encoding="utf-8"))
    current = state.get("current")
    if current in state["completed"]:
        next_action = "Requested steps completed; inspect their artifacts/report or continue with a later --through"
    elif current == "H5":
        next_action = f"Review {work / 'frozen.yaml'} and {work / 'protocol.json'}, then explicitly approve-h5"
    elif current in {"H1", "H2", "H3", "H4"}:
        next_action = f"Run operator review-info --work {work} --gate {current}; label samples, approve-review, resume"
    else:
        next_action = "Inspect the failed command/run metadata; do not overwrite partial outputs or retry final tests"
    return {"status": state["status"], "current": state.get("current"), "completed": list(state["completed"]),
            "next": next_action,
            "note": "A code-development assumption or mock test does not approve real experiments"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    inspect = commands.add_parser("status")
    inspect.add_argument("--work", type=Path, required=True)
    for name in ("review-info", "approve-review"):
        p = commands.add_parser(name)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--gate", choices=("H1", "H2", "H3", "H4"), required=True)
        p.add_argument("--schema", type=Path, default=Path("schemas/prime.yaml"))
        if name == "approve-review":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--yes", action="store_true")
    h5 = commands.add_parser("approve-h5")
    h5.add_argument("--plan", type=Path, required=True)
    h5.add_argument("--out", type=Path, required=True)
    h5.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "status":
            result = status(args.work)
        elif args.action == "review-info":
            result = review_evidence(args.gate, args.work, args.schema, validate=False)
        elif args.action == "approve-review":
            result = approve_review(args.work, args.gate, args.schema, args.out, confirmed=args.yes)
        else:
            result = approve_h5(args.plan, args.out, confirmed=args.yes)
    except (ReviewRequired, ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"{exc}\nNo approval was written.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
