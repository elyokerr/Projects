-- One row per local authority; pick the lexicographically-first region code
-- when the snapshot has inconsistent LA→region mappings (fixture artefact).
SELECT
    local_authority_code,
    MIN(region_code) AS region_code
FROM {{ ref('int_nspl_current') }}
WHERE local_authority_code IS NOT NULL
GROUP BY local_authority_code
