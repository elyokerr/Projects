SELECT DISTINCT
    postcode,
    lsoa_code,
    local_authority_code,
    region_code,
    imd_decile,
    latitude,
    longitude
FROM {{ ref('int_nspl_current') }}
