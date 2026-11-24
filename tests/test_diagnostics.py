import pandas as pd
import pytest

from slotspace.diagnostics import pacc_diagnostic


def test_pacc_diagnostic_fits_rates_and_measures_counts_on_disjoint_documents():
    rows = pd.DataFrame({"doc_id": list(range(8)), "slot": ["role"] * 8,
                         "native": [True, True, False, False, False, False, False, False],
                         "pi": [0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]})
    result = pacc_diagnostic(rows)
    assert result["raw_relative_error"] == pytest.approx(0.4)
    assert result["pacc_relative_error"] == pytest.approx(0)
    assert result["slots"]["role"]["n_calibration"] == result["slots"]["role"]["n_holdout"] == 4
