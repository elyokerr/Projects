import pyarrow.parquet as pq

from src.housing_mds.parquet_io import csv_to_parquet


def test_csv_to_parquet_preserves_schema(tmp_path):
    csv = tmp_path / "in.csv"
    csv.write_text("a,b,c\n1,2.5,hello\n3,4.5,world\n")
    out = tmp_path / "out.parquet"
    csv_to_parquet(csv, out, dtypes={"a": "int64", "b": "float64", "c": "string"})
    df = pq.read_table(out).to_pandas()
    assert list(df.columns) == ["a", "b", "c"]
    assert df["a"].tolist() == [1, 3]
    assert df["c"].tolist() == ["hello", "world"]


def test_csv_to_parquet_with_no_header(tmp_path):
    csv = tmp_path / "in.csv"
    csv.write_text('"id1",10\n"id2",20\n')
    out = tmp_path / "out.parquet"
    csv_to_parquet(
        csv, out,
        column_names=["uid", "price"],
        dtypes={"uid": "string", "price": "int64"},
        header=None,
    )
    df = pq.read_table(out).to_pandas()
    assert df["uid"].tolist() == ["id1", "id2"]
