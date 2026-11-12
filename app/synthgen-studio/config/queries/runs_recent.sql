-- Recent pipeline runs for the history rail. :refresh_bucket busts the 1h analytics cache (pass a minute bucket).
-- @param uc_schema STRING = main.synthgen_3d
-- @param refresh_bucket INT = 0
SELECT job_run_id, submitted_by, submitted_at, target_class_name, status, num_views,
       map50, precision, recall, model_version, source_image_path, updated_at
FROM IDENTIFIER(:uc_schema || '.pipeline_runs')
WHERE :refresh_bucket >= 0
ORDER BY submitted_at DESC NULLS LAST
LIMIT 25
