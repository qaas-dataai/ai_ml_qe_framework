import numpy as np
from scipy.stats import norm
import pytest

def test_performance_threshold():
    data = np.random.normal(50, 10, 1000)
    threshold = norm.ppf(0.95, np.mean(data), np.std(data))
    assert threshold > 60, "Performance threshold too low"
