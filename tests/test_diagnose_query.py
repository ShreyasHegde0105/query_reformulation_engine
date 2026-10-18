from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from evaluation.diagnose_query import (
    diagnose_query,
    get_query_reformulations,
    trace_semantic_filtering,
    print_diagnostic_report,
)
from evaluation.schema import GoldenQuery


def test_get_query_reformulations():
    reformulations = get_query_reformulations("Q004", "academic transcript")
    assert isinstance(reformulations, list)


@pytest.mark.integration
def test_diagnose_query_q004():
    diag = diagnose_query("Q004")
    assert diag["query_id"] == "Q004"
    assert "raw_query" in diag
    assert "normalized_query" in diag
    assert "relevant_chunks" in diag
    assert "filtered_queries" in diag
    assert "ground_truth_tracking" in diag
    assert "final_top_10" in diag


@pytest.mark.integration
def test_print_diagnostic_report(capsys):
    diag = diagnose_query("Q004")
    print_diagnostic_report(diag)
    captured = capsys.readouterr()
    assert "DIAGNOSTIC TRACE FOR QUERY: Q004" in captured.out
    assert "GROUND TRUTH RETRIEVAL TRACKING:" in captured.out
