-- Dataset composition for the model card.
-- @param uc_schema STRING = main.synthgen_3d
-- @param job_run_id BIGINT = 0
-- @param refresh_bucket INT = 0
SELECT split, background_kind, COUNT(*) AS n, AVG(mask_coverage) AS avg_coverage
FROM IDENTIFIER(:uc_schema || '.synthetic_views')
WHERE job_run_id = :job_run_id AND :refresh_bucket >= 0
GROUP BY split, background_kind
ORDER BY split, background_kind
