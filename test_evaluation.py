import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evaluation import (
    evaluate_retrieval,
    normalize_text,
    overlap_score,
    precision_at_k,
    recall_at_k,
    tokenize,
)

REPORT_PATH = Path(__file__).with_name("report.json")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("  Diabetes Research   Text! ", "diabetes research text"),
        ("Type 1 Diabetes", "type 1 diabetes"),
        ("CGM and HbA1c", "cgm and hba1c"),
        ("A   B\nC\tD", "a b c d"),
        ("Pediatric Diabetes Care.", "pediatric diabetes care"),
        ("   ", ""),
        ("Insulin Resistance 2024!", "insulin resistance 2024"),
        ("adolescents with type 2 diabetes", "adolescents with type 2 diabetes"),
        ("Hypoglycemia-risk management", "hypoglycemia risk management"),
        ("README.md", "readme md"),
    ],
)
def test_normalize_text(text, expected):
    assert normalize_text(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Diabetes care", ["diabetes", "care"]),
        ("Type 1 diabetes", ["type", "1", "diabetes"]),
        ("CGM and HbA1c", ["cgm", "and", "hba1c"]),
        ("pediatric\ncare", ["pediatric", "care"]),
        ("insulin-resistance", ["insulin", "resistance"]),
        ("A   B\tC", ["a", "b", "c"]),
        ("Hypoglycemia risk! management?", ["hypoglycemia", "risk", "management"]),
        ("2019 study of adolescents", ["2019", "study", "of", "adolescents"]),
        ("Type-2 diabetes", ["type", "2", "diabetes"]),
        ("", []),
    ],
)
def test_tokenize(text, expected):
    assert tokenize(text) == expected


@pytest.mark.parametrize(
    "query, document, expected",
    [
        ("diabetes", "diabetes", 1.0),
        ("diabetes care", "care diabetes", 1.0),
        ("diabetes care", "diabetes", 0.5),
        ("diabetes care", "insulin therapy", 0.0),
        ("type 1 diabetes", "type 1 diabetes management", 0.75),
        ("adolescents diabetes", "adolescents and diabetes", 2 / 3),
        ("hypoglycemia management", "hypoglycemia monitoring", 1 / 3),
        ("glucose monitoring", "continuous glucose monitoring", 0.6666666667),
        ("pediatric psychology", "psychology in pediatrics", 0.25),
        ("", "diabetes", 0.0),
    ],
)
def test_overlap_score(query, document, expected):
    assert overlap_score(query, document) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "relevant, retrieved, k, expected",
    [
        ({"A", "B"}, ["A", "C", "B"], 3, 2 / 3),
        ({"A", "B"}, ["A", "B", "C"], 2, 1.0),
        ({"A", "B"}, ["C", "D", "E"], 3, 0.0),
        ({"A"}, ["A", "B"], 1, 1.0),
        ({"A", "B"}, ["A"], 2, 0.5),
        ({"A", "B", "C"}, ["B", "A", "D"], 2, 1.0),
        ({"A", "B", "C"}, ["D", "E", "F"], 3, 0.0),
        ({"A", "B"}, ["A", "B", "C", "D"], 4, 0.5),
        ({"A", "B"}, ["B", "A"], 10, 0.2),
        ({}, ["A", "B"], 3, 0.0),
    ],
)
def test_precision_at_k(relevant, retrieved, k, expected):
    assert precision_at_k(relevant, retrieved, k) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "relevant, retrieved, k, expected",
    [
        ({"A", "B"}, ["A", "C", "B"], 3, 1.0),
        ({"A", "B"}, ["A", "B", "C"], 2, 1.0),
        ({"A", "B"}, ["C", "D", "E"], 3, 0.0),
        ({"A"}, ["A", "B"], 1, 1.0),
        ({"A", "B"}, ["A"], 2, 0.5),
        ({"A", "B", "C"}, ["B", "A", "D"], 2, 2 / 3),
        ({"A", "B", "C"}, ["D", "E", "F"], 3, 0.0),
        ({"A", "B"}, ["A", "B", "C", "D"], 4, 1.0),
        ({"A", "B"}, ["B", "A"], 10, 1.0),
        ({}, ["A", "B"], 3, 0.0),
    ],
)
def test_recall_at_k(relevant, retrieved, k, expected):
    assert recall_at_k(relevant, retrieved, k) == pytest.approx(expected, abs=1e-9)


def build_evaluation_report():
    """Create a summary report that mirrors the retrieval evaluation checks."""
    sample_cases = [
        {
            "query": "diabetes",
            "retrieved_docs": ["diabetes summary", "insulin", "care"],
            "relevant_docs": ["diabetes summary"],
            "top_k": 3,
        },
        {
            "query": "type 1 diabetes",
            "retrieved_docs": ["type 1 diabetes", "control", "education"],
            "relevant_docs": ["type 1 diabetes"],
            "top_k": 3,
        },
        {
            "query": "adolescents",
            "retrieved_docs": ["adolescent support", "care", "monitoring"],
            "relevant_docs": ["adolescent support"],
            "top_k": 3,
        },
        {
            "query": "care",
            "retrieved_docs": ["care", "monitoring", "self management"],
            "relevant_docs": ["care", "monitoring"],
            "top_k": 3,
        },
    ]

    results = []
    for case in sample_cases:
        report = evaluate_retrieval(
            case["query"],
            case["retrieved_docs"],
            case["relevant_docs"],
            case["top_k"],
        )
        results.append({
            "query": report["query"],
            "retrieved_docs": case["retrieved_docs"],
            "relevant_docs": case["relevant_docs"],
            "top_k": case["top_k"],
            "metrics": {
                "precision_at_k": report["precision_at_k"],
                "recall_at_k": report["recall_at_k"],
                "f1_score": report["f1_score"],
                "mrr": report["mrr"],
                "ndcg_at_k": report["ndcg_at_k"],
            },
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": "test_evaluation",
        "total_cases": len(results),
        "results": results,
    }


def write_report_json():
    report = build_evaluation_report()
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return str(REPORT_PATH)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Save the evaluation metrics to report.json whenever pytest runs this module."""
    if str(session.config.args[0]).endswith("test_evaluation.py"):
        path = write_report_json()
        print(f"\nEvaluation report saved to: {path}")


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-q"])
    report_path = write_report_json()
    print(f"\nEvaluation report saved to: {report_path}")
    raise SystemExit(exit_code)
