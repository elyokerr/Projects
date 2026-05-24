-- NSPL is a snapshot; "current" is the most recent loaded snapshot.
-- For now use it as the canonical postcode→geo lookup. SCD2 over snapshots is in the v2 backlog.
SELECT * FROM {{ ref('stg_nspl') }}
