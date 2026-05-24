-- Dedupe by postcode: NSPL fixture contains duplicate postcodes; take the
-- lexicographically-first lsoa11 to make the snapshot unique on postcode.
SELECT
    UPPER(pcd) AS postcode,
    lsoa11 AS lsoa_code,
    lad22cd AS local_authority_code,
    rgn AS region_code,
    lat AS latitude,
    long AS longitude,
    CAST(imd AS INTEGER) AS imd_decile
FROM {{ source('raw', 'nspl') }}
WHERE pcd IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY UPPER(pcd) ORDER BY lsoa11) = 1
