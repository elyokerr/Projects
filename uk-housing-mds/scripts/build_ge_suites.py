"""Create the three landing-zone expectation suites + aggregate checkpoint.

Idempotent. Run after build_ge_context.py.
"""

from __future__ import annotations

from pathlib import Path

import great_expectations as gx
from great_expectations.core import ExpectationConfiguration

ROOT = Path(__file__).resolve().parents[1]
GE_DIR = ROOT / "great_expectations"

ctx = gx.get_context(context_root_dir=str(GE_DIR))

PPD_COLUMNS = [
    "transaction_unique_id",
    "price_paid",
    "date_of_transfer",
    "postcode",
    "property_type",
    "new_build_flag",
    "tenure",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category_type",
    "record_status",
]

# NSPL fixture uses region codes (E12000001..N99999999), not names. Keep
# expectation aligned with what landing parquet actually contains.
NSPL_REGION_CODES = [
    "E12000001", "E12000002", "E12000003", "E12000004", "E12000005",
    "E12000006", "E12000007", "E12000008", "E12000009",
    "W99999999", "S99999999", "N99999999",
    None,
]

HPI_REGION_NAMES = [
    "North East", "North West", "Yorkshire and The Humber",
    "East Midlands", "West Midlands", "East of England", "London",
    "South East", "South West", "Wales", "Scotland", "Northern Ireland",
    "United Kingdom", "Great Britain", "England", "England and Wales",
]

POSTCODE_REGEX = r"^([A-Z]{1,2}[0-9][A-Z0-9]?) ([0-9][A-Z]{2})$"


def _ensure_suite(name: str, expectations: list[ExpectationConfiguration]) -> None:
    if name in [s for s in ctx.list_expectation_suite_names()]:
        ctx.delete_expectation_suite(name)
    suite = ctx.add_expectation_suite(expectation_suite_name=name)
    for e in expectations:
        suite.add_expectation(e)
    ctx.save_expectation_suite(suite)
    print(f"suite saved: {name} ({len(expectations)} expectations)")


_ensure_suite(
    "ppd_landing",
    [
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={"column_list": PPD_COLUMNS},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "price_paid", "type_": "int64"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "price_paid", "min_value": 1, "max_value": 100_000_000},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={"column": "postcode", "regex": POSTCODE_REGEX, "mostly": 0.99},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "transaction_unique_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "transaction_unique_id"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": 0, "max_value": 1_000_000_000},
        ),
    ],
)

_ensure_suite(
    "nspl_landing",
    [
        # Note: real NSPL has unique pcd, but the synthetic test fixture
        # duplicates rows to keep the file small. Uniqueness is re-asserted
        # downstream by the dbt source test on raw.nspl.pcd.
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "pcd"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "lsoa11", "mostly": 0.99},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "rgn", "value_set": NSPL_REGION_CODES},
        ),
    ],
)

_ensure_suite(
    "hpi_landing",
    [
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "region_name", "value_set": HPI_REGION_NAMES},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "average_price", "min_value": 1, "max_value": 10_000_000},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "date"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "area_code"},
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "region_name"},
        ),
    ],
)

# Build aggregate checkpoint that runs all three suites.
checkpoint_name = "landing_all"
validations = [
    {
        "batch_request": {
            "datasource_name": "landing_ppd",
            "data_asset_name": "ppd",
        },
        "expectation_suite_name": "ppd_landing",
    },
    {
        "batch_request": {
            "datasource_name": "landing_nspl",
            "data_asset_name": "nspl",
        },
        "expectation_suite_name": "nspl_landing",
    },
    {
        "batch_request": {
            "datasource_name": "landing_hpi",
            "data_asset_name": "hpi",
        },
        "expectation_suite_name": "hpi_landing",
    },
]

ctx.add_or_update_checkpoint(
    name=checkpoint_name,
    validations=validations,
    action_list=[
        {
            "name": "store_validation_result",
            "action": {"class_name": "StoreValidationResultAction"},
        },
        {
            "name": "update_data_docs",
            "action": {"class_name": "UpdateDataDocsAction"},
        },
    ],
)
print(f"checkpoint saved: {checkpoint_name}")
