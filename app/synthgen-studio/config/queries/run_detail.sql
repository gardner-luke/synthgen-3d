-- Full provenance row for one run.
-- @param uc_schema STRING = main.synthgen_3d
-- @param job_run_id BIGINT = 0
-- @param refresh_bucket INT = 0
SELECT job_run_id, submitted_by, submitted_at, source_image_path, source_sha256, target_class_name, params,
       status, ply_path, gaussian_count, idol_seconds, num_views, dataset_path, render_seconds,
       mlflow_run_id, model_name, model_version, map50, map50_95, precision, recall, train_seconds,
       real_test_image_path, real_test_max_conf, updated_at
FROM IDENTIFIER(:uc_schema || '.pipeline_runs')
WHERE job_run_id = :job_run_id AND :refresh_bucket >= 0
