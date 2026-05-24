"""UK postcode normalisation. Inward part is always 3 chars (NUM ALPHA ALPHA)."""

from __future__ import annotations

import re

_POSTCODE_RE = re.compile(
    r"^([A-Z]{1,2}[0-9][A-Z0-9]?)\s*([0-9][A-Z]{2})$"
)


def normalise_postcode(raw: str | None) -> str | None:
    if not raw:
        return None
    compact = re.sub(r"\s+", "", str(raw)).upper()
    if len(compact) < 5:
        return None
    inward = compact[-3:]
    outward = compact[:-3]
    candidate = f"{outward} {inward}"
    if _POSTCODE_RE.match(candidate):
        return candidate
    return None


def is_valid_postcode(raw: str | None) -> bool:
    return normalise_postcode(raw) is not None
