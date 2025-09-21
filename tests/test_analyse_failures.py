import pandas as pd
import pytest
import json
import re

INPUT_FILE = "demo-eval.xlsx"
OUTPUT_FILE = "result.csv"

def extract_failure_reason(row):
    chat = str(row.get("chat_history", ""))

    # Detect errors in chat history
    chat_lower = chat.lower()
    if "error" in chat_lower or "exception" in chat_lower:
        return {
            "failure_reason": "Error/Exception found in chat history",
            "mismatch_reason": None,
            "category_mismatch": None,
            "chat_history_error": True,
            "chat_history_error_detail": chat
        }

    # Check response JSON
    try:
        resp = json.loads(row.get("response", "{}"))
        value_text = resp.get("value", "")
        if "Mismatched Values" in value_text:
            mismatches = value_text.split("Mismatched Values:")[-1].strip()
            match = re.search(r"([a-zA-Z0-9_]+)\s*:", mismatches)
            category = match.group(1) if match else "Unknown"
            return {
                "failure_reason": "Mismatched values detected",
                "mismatch_reason": mismatches,
                "category_mismatch": category,
                "chat_history_error": False,
                "chat_history_error_detail": "No error"
            }
        return {
            "failure_reason": "Failure (reason not clearly specified in response)",
            "mismatch_reason": None,
            "category_mismatch": None,
            "chat_history_error": False,
            "chat_history_error_detail": "No error"
        }
    except Exception as e:
        return {
            "failure_reason": f"Response parsing error: {e}",
            "mismatch_reason": None,
            "category_mismatch": None,
            "chat_history_error": False,
            "chat_history_error_detail": "No error"
        }

@pytest.fixture(scope="module")
def zero_score_df():
    df = pd.read_excel(INPUT_FILE)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # Force numeric conversion for score
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    zero_df = df[df["score"] == 0].copy()

    # Apply extractor
    results = zero_df.apply(extract_failure_reason, axis=1, result_type="expand")
    zero_df = pd.concat([zero_df, results], axis=1)

    # Save results with summary
    if not zero_df.empty:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            zero_df.to_csv(f, index=False)
            f.write("\n\n=== Failure Counts by Category ===\n")
            summary = zero_df["category_mismatch"].value_counts()
            summary.to_csv(f, header=["count"])

    return zero_df

def test_zero_score_analysis(zero_score_df):
    print("\n=== Zero Score Failure Analysis ===")

    count = len(zero_score_df)
    print(f"Found {count} zero-score records")

    if zero_score_df.empty:
        pytest.skip("No zero-score cases found.")
    else:
        print(zero_score_df.to_string(index=False))

        # Print summary in console
        summary = zero_score_df["category_mismatch"].value_counts()
        print("\n=== Failure Counts by Category ===")
        print(summary.to_string())

        # Assert all scores are indeed zero
        assert (zero_score_df["score"] == 0).all(), "Non-zero score found in filtered records"
