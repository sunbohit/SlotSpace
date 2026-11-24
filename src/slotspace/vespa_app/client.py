"""Small explicit HTTP client for a dedicated, self-hosted Vespa instance."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice
import json
import logging
from pathlib import Path
from time import monotonic, perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .codec import document_number

logger = logging.getLogger(__name__)


class VespaError(RuntimeError):
    pass


class VespaUnavailable(VespaError):
    pass


def _endpoint(value: str) -> str:
    parts = urlsplit(value)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.path not in {"", "/"}
            or parts.query or parts.fragment or parts.username or parts.password):
        raise ValueError("Use an http(s) service origin without paths, credentials or query parameters")
    return value.rstrip("/")


def _scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise VespaError(f"Vespa returned a missing/nonfinite {name}")
    return float(value)


def parse_hits(response: dict, *, binding: bool) -> list[dict]:
    root = response.get("root")
    if not isinstance(root, dict):
        raise VespaError("Search response has no root")
    coverage = root.get("coverage", {})
    if coverage.get("coverage") != 100 or coverage.get("full") is False or coverage.get("degraded"):
        raise VespaError(f"Partial/degraded search coverage: {coverage}")
    hits = []
    for hit in root.get("children", []):
        fields = hit.get("fields", {})
        features = fields.get("matchfeatures", {})
        if "doc_id" not in fields:
            raise VespaError("A returned hit has no document ID (unexpected hit group)")

        def feature(name):
            key = name if name in features else f"rankingExpression({name})"
            return _scalar(features.get(key), name)

        hits.append({"doc_id": document_number(fields["doc_id"]),
                     "bind": feature("bind") if binding else 0.0,
                     "dense": feature("dense"), "bm25": feature("bm25(text)")})
    if len(hits) != len({h["doc_id"] for h in hits}):
        raise VespaError("Vespa returned duplicate document IDs")
    return hits


class VespaClient:
    def __init__(self, endpoint="http://localhost:8080", deploy_endpoint="http://localhost:19071", timeout=10.0):
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("HTTP timeout must be positive")
        self.endpoint, self.deploy_endpoint, self.timeout = _endpoint(endpoint), _endpoint(deploy_endpoint), timeout

    def version(self) -> str:
        version = self._request(self.endpoint, "/state/v1/version").get("version")
        if not isinstance(version, str) or not version.strip():
            raise VespaError("Vespa service did not identify its version")
        return version

    def _request(self, origin, path, *, method="GET", body=None, content_type="application/json", timeout=None):
        request = Request(origin + path, data=body, method=method, headers={"Content-Type": content_type})
        try:
            with urlopen(request, timeout=self.timeout if timeout is None else timeout) as src:
                raw = src.read()
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            error = VespaUnavailable if exc.code in {502, 503, 504} else VespaError
            raise error(f"Vespa HTTP {exc.code} at {path}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise VespaUnavailable(f"Vespa request failed at {origin}{path}: {exc}") from exc
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise VespaError(f"Vespa returned invalid JSON at {path}") from exc
        if not isinstance(result, dict):
            raise VespaError(f"Vespa returned a non-object response at {path}")
        root = result.get("root", {})
        if not isinstance(root, dict):
            raise VespaError(f"Vespa returned an invalid root at {path}")
        if result.get("error-code") or result.get("errors") or root.get("errors"):
            raise VespaError(f"Vespa reported errors at {path}: {result}")
        return result

    def wait_ready(self, timeout=300.0):
        deadline = monotonic() + timeout
        while True:
            try:
                health = self._request(self.endpoint, "/state/v1/health")
                if health.get("status", {}).get("code") == "up":
                    return
                logger.info("Waiting for Vespa health: %s", health)
            except VespaUnavailable as exc:
                logger.info("Waiting for Vespa: %s", exc)
            if monotonic() >= deadline:
                raise VespaUnavailable("Vespa did not become healthy before the readiness deadline")
            sleep(1)

    def deploy(self, app: Path):
        required = ("services.xml", "hosts.xml", "schemas/doc.sd")
        archive = BytesIO()
        with ZipFile(archive, "w", ZIP_DEFLATED) as zipped:
            for name in required:
                zipped.writestr(name, app.joinpath(*name.split("/")).read_bytes())
        response = self._request(self.deploy_endpoint, "/application/v2/tenant/default/prepareandactivate",
                                 method="POST", body=archive.getvalue(), content_type="application/zip", timeout=300)
        if "session-id" not in response:
            raise VespaError("Deployment response lacks a session ID")
        if any(row.get("level") in {"ERROR", "SEVERE"} for row in response.get("log", [])):
            raise VespaError(f"Deployment reported errors: {response['log']}")
        actions = response.get("configChangeActions", {})
        if any(actions.get(k) for k in ("refeed", "restart", "reindex")):
            raise VespaError(f"Deployment requires explicit operational action: {actions}")
        self.wait_ready()
        return response

    def feed_one(self, operation: dict):
        number = document_number(operation["fields"]["doc_id"])
        if operation["put"] != f"id:slotspace:doc::{number}":
            raise ValueError("Feed operation ID does not match its document")
        return self._request(self.endpoint, f"/document/v1/slotspace/doc/docid/{number}", method="POST",
                             body=json.dumps({"fields": operation["fields"]}, allow_nan=False).encode("utf-8"))

    def delete_one(self, doc_id):
        number = document_number(doc_id)
        return self._request(self.endpoint, f"/document/v1/slotspace/doc/docid/{number}", method="DELETE")

    def feed(self, operations, *, workers=8):
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("Feed concurrency must lie in [1, 32]")
        start, count = perf_counter(), 0
        iterator = iter(operations)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while batch := list(islice(iterator, workers * 4)):
                for _ in pool.map(self.feed_one, batch):
                    count += 1
        if not count:
            raise ValueError("Refusing to report an empty feed as successful")
        return {"documents": count, "feed_ms": 1000 * (perf_counter() - start)}

    def search(self, body: dict):
        start = perf_counter()
        response = self._request(self.endpoint, "/search/", method="POST",
                                 body=json.dumps(body, allow_nan=False).encode("utf-8"))
        elapsed = 1000 * (perf_counter() - start)
        features = parse_hits(response, binding=body["ranking"] != "hybrid")
        search_ms = 1000 * _scalar(response.get("timing", {}).get("searchtime"), "searchtime")
        if search_ms < 0:
            raise VespaError("Negative server searchtime")
        return {"hits": features, "server_ms": search_ms, "rpc_ms": elapsed}

    def document_count(self):
        response = self._request(self.endpoint, "/search/", method="POST", body=json.dumps({
            "yql": "select * from doc where true", "hits": 0, "timeout": "5s",
            "ranking.softtimeout.enable": False,
        }).encode())
        parse_hits(response, binding=False)
        count = response["root"].get("fields", {}).get("totalCount")
        if type(count) is not int or count < 0:
            raise VespaError("Search response has no valid total document count")
        return count

    def wait_count(self, expected: int, timeout=300.0):
        deadline = monotonic() + timeout
        while True:
            count = self.document_count()
            if count == expected:
                return
            if count > expected:
                raise VespaError("Dedicated engine contains unexpected extra documents")
            if monotonic() >= deadline:
                raise VespaUnavailable(f"Only {count}/{expected} fed documents became searchable")
            sleep(0.5)
