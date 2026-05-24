"""Cron-schedule reference for the monthly refresh.

This constant is reference-only. Actual scheduling lives in
``.github/workflows/monthly-refresh.yml``; Prefect deployments are not used in
this project's free-tier setup.
"""

from __future__ import annotations

MONTHLY_CRON = "0 6 1 * *"
"""06:00 UTC on the first of every month."""
