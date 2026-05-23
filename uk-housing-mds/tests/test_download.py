import zipfile
from unittest.mock import MagicMock, patch

from src.housing_mds.download import download_file, verify_csv_magic, verify_zip_magic


def test_download_writes_to_path(tmp_path):
    target = tmp_path / "out.csv"
    fake_resp = MagicMock(status_code=200)
    fake_resp.iter_content = lambda chunk_size: [b"col_a,col_b\n", b"1,2\n"]
    fake_resp.raise_for_status = MagicMock()
    with patch("src.housing_mds.download.requests.get", return_value=fake_resp):
        download_file("http://x/y.csv", target)
    assert target.read_text() == "col_a,col_b\n1,2\n"


def test_download_uses_browsery_user_agent(tmp_path):
    target = tmp_path / "out.csv"
    fake_resp = MagicMock(status_code=200)
    fake_resp.iter_content = lambda chunk_size: [b"x"]
    fake_resp.raise_for_status = MagicMock()
    with patch("src.housing_mds.download.requests.get", return_value=fake_resp) as get:
        download_file("http://x/y.csv", target)
    headers = get.call_args.kwargs["headers"]
    assert "Mozilla/5.0" in headers["User-Agent"]


def test_download_skips_if_exists_and_force_false(tmp_path):
    target = tmp_path / "out.csv"
    target.write_text("already here")
    with patch("src.housing_mds.download.requests.get") as get:
        download_file("http://x/y.csv", target, force=False)
    assert get.call_count == 0
    assert target.read_text() == "already here"


def test_verify_csv_magic_accepts_text(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("transaction_unique_id,price_paid\n")
    assert verify_csv_magic(p) is True


def test_verify_csv_magic_rejects_html_error_page(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("<!doctype html><html>404</html>")
    assert verify_csv_magic(p) is False


def test_verify_zip_magic_accepts_real_zip(tmp_path):
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("inner.csv", "a,b\n1,2\n")
    assert verify_zip_magic(p) is True
