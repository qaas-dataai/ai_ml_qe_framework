import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pytest

def test_log_anomalies():
    logs = ["ERROR: Database connection failed", "WARN: High memory usage", "ERROR: API timeout reached", "INFO: User login successful"]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(logs)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    assert len(set(kmeans.labels_)) == 2, "Log clustering failed"
