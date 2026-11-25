from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from threading import Thread
from zipfile import ZipFile

import pytest

from slotspace.vespa_app.application import write_application
from slotspace.vespa_app.application import EXPONENTS, profile_name
from slotspace.vespa_app.client import VespaClient, VespaError, VespaUnavailable, parse_hits
from slotspace.vespa_app.smoke import smoke, smoke_fixture
from slotspace.planner import plan_readings
from slotspace.reference import ScoreConfig, binding_score


@pytest.fixture
def server():
    replies, requests = deque(), []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.handle_request()

        def do_POST(self):
            self.handle_request()

        def do_DELETE(self):
            self.handle_request()

        def handle_request(self):
            requests.append((self.command, self.path, self.rfile.read(int(self.headers.get("Content-Length", 0)))))
            status, body = replies.popleft()
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=service.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{service.server_port}"
        yield VespaClient(endpoint, endpoint), replies, requests
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=5)


def _response(features=None):
    return {"timing": {"searchtime": 0.002},
            "root": {"coverage": {"coverage": 100}, "children": [
                {"fields": {"doc_id": 17, "matchfeatures": features or {
                    "rankingExpression(bind)": 0.3, "rankingExpression(dense)": 0.2, "bm25(text)": 0.1}}}]}}


def test_http_query_preserves_named_features_and_reports_real_server_and_rpc_time(server):
    client, replies, requests = server
    replies.append((200, _response()))
    result = client.search({"ranking": "slotspace", "query": "example"})
    assert result["hits"] == [{"doc_id": 17, "bind": 0.3, "dense": 0.2, "bm25": 0.1}]
    assert result["server_ms"] == 2 and result["rpc_ms"] >= 0
    assert requests[0][:2] == ("POST", "/search/")
    assert json.loads(requests[0][2])["query"] == "example"
    replies.append((200, {"version": "8.fixture"}))
    assert client.version() == "8.fixture" and requests[-1][1] == "/state/v1/version"


def test_http_success_cannot_hide_query_errors_partial_results_or_bad_feature_values(server):
    client, replies, _ = server
    error = _response()
    error["root"]["errors"] = [{"code": 12, "message": "Timed out"}]
    replies.append((200, error))
    with pytest.raises(VespaError, match="Timed out"):
        client.search({"ranking": "slotspace"})
    replies.append((503, {"message": "starting"}))
    with pytest.raises(VespaUnavailable, match="503"):
        client.document_count()
    replies.append((200, b"not-json"))
    with pytest.raises(VespaError, match="invalid JSON"):
        client.document_count()
    bad = _response()
    bad["root"]["coverage"] = {"coverage": 100, "degraded": {"timeout": True}}
    with pytest.raises(VespaError, match="degraded"):
        parse_hits(bad, binding=True)
    with pytest.raises(VespaError, match="bind"):
        parse_hits(_response({"dense": 1, "bm25(text)": 0}), binding=True)


def test_deploy_uploads_root_relative_app_and_uses_document_api_not_feed_envelopes(tmp_path, server):
    client, replies, requests = server
    app = tmp_path / "app"
    write_application(app)
    replies.extend([(200, {"session-id": "1"}), (200, {"status": {"code": "up"}})])
    client.deploy(app)
    assert requests[0][:2] == ("POST", "/application/v2/tenant/default/prepareandactivate")
    with ZipFile(BytesIO(requests[0][2])) as zipped:
        assert set(zipped.namelist()) == {"services.xml", "hosts.xml", "schemas/doc.sd"}
        assert b"slotspace_a0_5_b2" in zipped.read("schemas/doc.sd")
    op = smoke_fixture()[0][0]
    replies.append((200, {"id": op["put"]}))
    client.feed_one(op)
    assert requests[-1][:2] == ("POST", "/document/v1/slotspace/doc/docid/1")
    assert json.loads(requests[-1][2]) == {"fields": op["fields"]}
    replies.append((200, {"id": op["put"]}))
    client.delete_one(1)
    assert requests[-1][:2] == ("DELETE", "/document/v1/slotspace/doc/docid/1")


def _count_response(count):
    return {"root": {"coverage": {"coverage": 100}, "fields": {"totalCount": count}}}


@pytest.mark.parametrize("fault", [None, "query-error", "parity-error"])
def test_mock_http_smoke_runs_deploy_feed_all_profiles_and_cleans_up_even_on_failure(server, fault):
    client, replies, requests = server
    _, reference, plan, _, _ = smoke_fixture()
    readings = plan_readings(plan)
    health = {"status": {"code": "up"}}
    replies.extend([(200, {"session-id": "synthetic"}), (200, health), (200, health),
                    (200, _count_response(0)), *[(200, {"id": "synthetic"}) for _ in reference],
                    (200, _count_response(4))])
    checked = EXPONENTS if fault is None else EXPONENTS[:2]
    for a, b in checked:
        response = {"timing": {"searchtime": 0.001}, "root": {"coverage": {"coverage": 100}, "children": [
            {"fields": {"doc_id": number, "matchfeatures": {
                "rankingExpression(bind)": binding_score(doc, readings, ScoreConfig(a=a, b=b, precision="binary")),
                "rankingExpression(dense)": 0.2, "bm25(text)": 0.1}}} for number, doc in reference.items()]}}
        if fault and (a, b) == checked[-1]:
            if fault == "query-error":
                response["root"]["errors"] = [{"message": "synthetic ranking failure"}]
            else:
                response["root"]["children"][0]["fields"]["matchfeatures"]["rankingExpression(bind)"] += 0.01
        replies.append((200, response))
    if fault is None:
        replies.append((200, response))
    replies.extend([*[(200, {"id": "synthetic"}) for _ in reference], (200, _count_response(0))])
    if fault:
        with pytest.raises(VespaError if fault == "query-error" else ValueError):
            smoke(client, deploy=True)
    else:
        result = smoke(client, deploy=True)
        assert result["feed"]["documents"] == 4
        assert all(v["candidates"] == 4 and v["max_abs_difference"] <= 1e-3 for v in result["profiles"].values())
    assert not replies
    assert requests[0][:2] == ("POST", "/application/v2/tenant/default/prepareandactivate")
    fed = [json.loads(body)["fields"]["doc_id"] for method, path, body in requests
           if method == "POST" and path.startswith("/document/v1/")]
    assert set(fed) == {1, 2, 3, 4}
    profiles = [json.loads(body)["ranking"] for method, path, body in requests
                if method == "POST" and path == "/search/" and "ranking" in json.loads(body)]
    assert profiles == [profile_name(a, b) for a, b in checked] + ([] if fault else ["hybrid"])
    assert {path for method, path, _ in requests if method == "DELETE"} == {
        f"/document/v1/slotspace/doc/docid/{i}" for i in reference}


def test_mock_http_smoke_refuses_to_touch_a_nonempty_engine(server):
    client, replies, requests = server
    replies.extend([(200, {"status": {"code": "up"}}), (200, _count_response(17))])
    with pytest.raises(ValueError, match="EMPTY"):
        smoke(client)
    assert not any(path.startswith("/document/v1/") for _, path, _ in requests)
