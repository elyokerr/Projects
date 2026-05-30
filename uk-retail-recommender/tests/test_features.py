import pandas as pd

from src.ranking.features import FEATURE_COLUMNS, build_features


def test_features_one_row_per_candidate_with_expected_columns():
    candidates = [{"item_id": "a", "pop": 0.5, "als": 1.0, "i2i": 0.0, "tt": 0.0},
                  {"item_id": "c", "pop": 0.1, "als": 0.0, "i2i": 0.3, "tt": 0.0}]
    item_meta = pd.DataFrame({"item_id": ["a", "c"], "price": [2.0, 4.0], "popularity": [10, 3]})
    cust = {"recency_days": 5, "frequency": 4, "monetary": 30.0, "country": "United Kingdom"}
    owned = {"a"}
    X = build_features(candidates, item_meta, cust, owned)
    assert len(X) == 2
    for col in ["pop", "als", "i2i", "tt", "price", "popularity",
                "recency_days", "frequency", "monetary", "prev_bought"]:
        assert col in X.columns
    assert X.set_index("item_id").loc["a", "prev_bought"] == 1


def test_feature_columns_order_matches_constant():
    """FEATURE_COLUMNS must be selectable from the output without KeyError."""
    candidates = [{"item_id": "x", "pop": 0.2, "als": 0.5, "i2i": 0.1, "tt": 0.0}]
    item_meta = pd.DataFrame({"item_id": ["x"], "price": [1.0], "popularity": [5]})
    cust = {"recency_days": 10, "frequency": 2, "monetary": 20.0, "country": "Germany"}
    owned: set = set()
    X = build_features(candidates, item_meta, cust, owned)
    # selecting FEATURE_COLUMNS must work and return exactly those columns
    selected = X[FEATURE_COLUMNS]
    assert list(selected.columns) == FEATURE_COLUMNS


def test_unknown_country_maps_to_minus_one():
    candidates = [{"item_id": "z", "pop": 0.0, "als": 0.0, "i2i": 0.0, "tt": 0.0}]
    item_meta = pd.DataFrame({"item_id": ["z"], "price": [1.0], "popularity": [1]})
    cust = {"recency_days": 1, "frequency": 1, "monetary": 1.0, "country": "Narnia"}
    X = build_features(candidates, item_meta, cust, set())
    assert X["country_code"].iloc[0] == -1


def test_not_owned_item_gets_prev_bought_zero():
    candidates = [{"item_id": "b", "pop": 0.3, "als": 0.0, "i2i": 0.0, "tt": 0.0}]
    item_meta = pd.DataFrame({"item_id": ["b"], "price": [3.0], "popularity": [7]})
    cust = {"recency_days": 3, "frequency": 3, "monetary": 15.0, "country": "France"}
    X = build_features(candidates, item_meta, cust, {"a"})
    assert X.set_index("item_id").loc["b", "prev_bought"] == 0
