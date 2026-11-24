"""Opt-in real service test. Never deploy into an existing/shared application."""

import os

import pytest

from slotspace.vespa_app.client import VespaClient
from slotspace.vespa_app.smoke import smoke


@pytest.mark.vespa
def test_real_vespa_binding_smoke_for_every_generated_profile():
    endpoint = os.environ.get("SLOTSPACE_VESPA_TEST_ENDPOINT")
    if not endpoint or os.environ.get("SLOTSPACE_VESPA_ALLOW_REPLACE") != "1":
        pytest.skip("Set a dedicated test endpoint and SLOTSPACE_VESPA_ALLOW_REPLACE=1 to deploy/feed real Vespa")
    deploy_endpoint = os.environ.get("SLOTSPACE_VESPA_DEPLOY_ENDPOINT", "http://localhost:19071")
    result = smoke(VespaClient(endpoint, deploy_endpoint), deploy=True)
    assert result["status"] == "ok" and result["source"] == "real Vespa HTTP"
    assert len(result["profiles"]) == 6
    assert all(p["candidates"] == 4 and p["max_abs_difference"] <= 1e-3 and p["order_inversions"] == 0
               for p in result["profiles"].values())
