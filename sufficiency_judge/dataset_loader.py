from pathlib import Path

import pandas as pd

DATASET_PATH = Path(__file__).resolve().with_name("cls.csv")

REQUIRED_COLUMNS = [
    "question",
    "answer",
    "sufficiency_label",
    "source_file"
]


def load_dataset(path=None):
    """Load the reference CSV, independent of the caller's working directory."""
    if path is None:
        path = DATASET_PATH
    df = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=[
        "question",
        "answer",
        "sufficiency_label"
    ])

    df["sufficiency_label"] = pd.to_numeric(
        df["sufficiency_label"],
        errors="raise"
    ).astype(int)

    if not df["sufficiency_label"].isin([0, 1]).all():
        raise ValueError("Labels must be either 0 or 1.")

    for column in ("question", "answer", "source_file"):
        df[column] = df[column].astype(str).str.strip()

    df = df[(df["question"] != "") & (df["answer"] != "")]

    return df


def prepare_reference_data(df, current_question):
    """
    Use the complete dataset except records belonging
    to the question currently being evaluated.
    """

    normalized_question = current_question.strip().casefold()

    reference_df = df[
        df["question"].astype(str).str.strip().str.casefold()
        != normalized_question
    ]

    return reference_df
