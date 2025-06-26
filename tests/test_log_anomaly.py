import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pytest
import re


def test_log_anomalies():
    logs = ["ERROR: Database connection failed", "WARN: High memory usage", "ERROR: API timeout reached", "INFO: User login successful"]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(logs)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    assert len(set(kmeans.labels_)) == 2, "Log clustering failed"

def test_log_clustering():
    """
    Cluster logs into 2 groups and validate clustering behavior.
    """
    logs = [
        "ERROR: Database connection failed",
        "WARN: High memory usage",
        "ERROR: API timeout reached",
        "INFO: User login successful"
    ]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(logs)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)

    assert len(set(kmeans.labels_)) == 2, "Log clustering failed"

def test_record_sync_count_anomaly():
    """
    Check if sync log indicates anomalously low record count.
    """
    log = "INFO: syncing records for current period close - no of records are 50000"
    match = re.search(r'no of records are (\d+)', log)
    assert match is not None, "No record count found in log"

    record_count = int(match.group(1))
    assert record_count >= 100000, f"❌ Anomaly detected: record count too low ({record_count})"

def test_record_sync_count_ok():
    """
    Pass case: record count is sufficient.
    """
    log = "INFO: syncing records for current period close - no of records are 150000"
    match = re.search(r'no of records are (\d+)', log)
    assert match is not None, "No record count found in log"

    record_count = int(match.group(1))
    assert record_count >= 100000, f"❌ Anomaly detected: record count too low ({record_count})"

def test_log_contains_errors():
    """
    Detect if logs contain any 'ERROR' level entries.
    """
    logs = [
        "INFO: job started",
        "INFO: syncing records",
        "ERROR: null pointer exception",
        "INFO: job completed"
    ]
    error_logs = [log for log in logs if "ERROR" in log]
    assert len(error_logs) > 0, "No error logs found (unexpected)"
