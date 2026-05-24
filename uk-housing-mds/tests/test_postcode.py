import pytest

from src.housing_mds.postcode import is_valid_postcode, normalise_postcode


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SW1A 1AA", "SW1A 1AA"),
        ("sw1a1aa", "SW1A 1AA"),
        ("sw1a 1aa", "SW1A 1AA"),
        ("  SW1A1AA ", "SW1A 1AA"),
        ("E1 6AN", "E1 6AN"),
        ("E16AN", "E1 6AN"),
        ("BT79 0NG", "BT79 0NG"),  # NI 4-char outward
    ],
)
def test_normalise(raw, expected):
    assert normalise_postcode(raw) == expected


def test_normalise_returns_none_for_garbage():
    assert normalise_postcode("XYZ") is None
    assert normalise_postcode("") is None
    assert normalise_postcode(None) is None


def test_is_valid_accepts_well_formed():
    assert is_valid_postcode("SW1A 1AA")
    assert is_valid_postcode("E1 6AN")
    assert is_valid_postcode("BT79 0NG")


def test_is_valid_rejects_malformed():
    assert not is_valid_postcode("SW1A1A")    # missing final letter
    assert not is_valid_postcode("123 456")
    assert not is_valid_postcode("")
