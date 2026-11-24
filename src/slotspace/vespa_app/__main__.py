"""Export, explicitly deploy/feed, smoke-test, or measure the optional Vespa engine."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import yaml

from ..artifacts import logged_run, read_jsonl, sha256, write_json
from .client import VespaClient
from .feed import verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    export = commands.add_parser("export", help="Package already computed research artifacts; does not run models")
    export.add_argument("--config", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    for action in ("feed", "smoke", "measure"):
        p = commands.add_parser(action)
        p.add_argument("--endpoint", default="http://localhost:8080")
        p.add_argument("--deploy-endpoint", default="http://localhost:19071")
        p.add_argument("--runs", type=Path, default=Path("runs"))
        if action in {"feed", "smoke"}:
            p.add_argument("--deploy", action="store_true", help="Replace the app on this dedicated deployment")
            p.add_argument("--allow-replace-application", action="store_true",
                           help="Explicitly acknowledge the deployment is disposable and dedicated")
        if action in {"feed", "measure"}:
            p.add_argument("--bundle", type=Path, required=True)
        if action == "feed":
            p.add_argument("--workers", type=int, default=8)
        if action == "measure":
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--feed-receipt", type=Path, required=True)
            p.add_argument("--queries", type=int, default=1000)
            p.add_argument("--parity-queries", type=int, default=200)
            p.add_argument("--device", default="cuda")
            p.add_argument("--container", help="Dedicated container name for docker stats; never starts/stops it")
    args = parser.parse_args()
    if args.action == "export":
        from .runtime import export_runtime

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        with logged_run(Path(config["paths"]["runs"]), "m9-export",
                        {"config_sha256": sha256(args.config), "out": str(args.out)}) as (run, metadata):
            export_runtime(config, args.out)
            metadata["bundle_manifest_sha256"] = sha256(args.out / "manifest.json")
        print(json.dumps({"bundle": str(args.out), "run": str(run)}))
        return
    if args.action in {"feed", "smoke"} and args.deploy and not args.allow_replace_application:
        parser.error("--deploy requires --allow-replace-application on a dedicated, disposable engine")
    client = VespaClient(args.endpoint, args.deploy_endpoint)
    if args.action == "measure":
        from .measure import run_measurements

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        print(run_measurements(config, args.bundle, args.feed_receipt, client, args.runs, queries=args.queries,
                               parity_queries=args.parity_queries, device=args.device, container=args.container))
        return
    with logged_run(args.runs, f"m9-{args.action}", {
        "endpoint": client.endpoint, "deploy_endpoint": client.deploy_endpoint, "deploy": args.deploy,
    }) as (run, metadata):
        if args.action == "smoke":
            from .smoke import smoke

            result = smoke(client, deploy=args.deploy)
            write_json(run / "smoke.json", result)
        else:
            manifest = verify_bundle(args.bundle)
            if args.deploy:
                metadata["deployment"] = client.deploy(args.bundle / "app")
            client.wait_ready()
            if client.document_count() != 0:
                raise ValueError("Feeding requires an empty dedicated app; no documents were overwritten")
            started = perf_counter()
            stats = client.feed(read_jsonl(args.bundle / "feed.jsonl"), workers=args.workers)
            if stats["documents"] != manifest["documents"]:
                raise ValueError("Fed document count differs from the verified bundle")
            client.wait_count(stats["documents"])
            visible_ms = 1000 * (perf_counter() - started)
            write_json(run / "feed.json", {**stats, "status": "ok", "endpoint": client.endpoint,
                                          "engine_version": client.version(), "feed_visible_ms": visible_ms,
                                          "application": manifest["application"],
                                          "feed_sha256": manifest["feed_sha256"]})
        print(run)


if __name__ == "__main__":
    main()
