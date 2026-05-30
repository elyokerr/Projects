import numpy as np
import pandas as pd

from src.ranking.ranker import FEATURE_COLUMNS, Ranker


def test_ranker_orders_positive_above_negative():
    # 2 customers (groups), each 3 candidates; the high-als item is the positive
    rows, y, groups = [], [], []
    for _ in range(2):
        for als, label in [(2.0, 1), (0.1, 0), (0.0, 0)]:
            rows.append({c: 0.0 for c in FEATURE_COLUMNS} | {"als": als})
            y.append(label)
        groups.append(3)
    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    r = Ranker().fit(X, np.array(y), groups)
    scores = r.predict(X.iloc[:3])
    assert scores.argmax() == 0  # the als=2.0 candidate ranks first


def test_ranker_predict_returns_array_of_correct_length():
    rows, y, groups = [], [], []
    for _ in range(3):
        for als, label in [(1.5, 1), (0.5, 0), (0.0, 0)]:
            rows.append({c: 0.0 for c in FEATURE_COLUMNS} | {"als": als})
            y.append(label)
        groups.append(3)
    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    r = Ranker().fit(X, np.array(y), groups)
    out = r.predict(X)
    assert len(out) == len(X)
    assert isinstance(out, np.ndarray)


def test_feature_columns_importable_from_ranker():
    """FEATURE_COLUMNS must be re-exportable from ranker.py."""
    from src.ranking.ranker import FEATURE_COLUMNS as FC
    assert isinstance(FC, list)
    assert "als" in FC
    assert "pop" in FC
    assert "prev_bought" in FC
    assert "country_code" in FC
