-- PPD joined to current postcode geography; resolves to LA + region.
SELECT
    p.transaction_unique_id,
    p.price_paid_gbp,
    p.transfer_date,
    p.postcode,
    p.property_type,
    p.is_new_build,
    p.tenure,
    p.district,
    p.county,
    n.lsoa_code,
    n.local_authority_code,
    n.region_code,
    n.imd_decile,
    n.latitude,
    n.longitude
FROM {{ ref('stg_ppd') }} p
LEFT JOIN {{ ref('int_nspl_current') }} n USING (postcode)
