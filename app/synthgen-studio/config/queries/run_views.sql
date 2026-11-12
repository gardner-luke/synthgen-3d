-- Gallery sample: a deterministic spread of rendered views with their bounding boxes.
-- @param uc_schema STRING = main.synthgen_3d
-- @param job_run_id BIGINT = 0
-- @param refresh_bucket INT = 0
SELECT view_id, split, image_path, bbox_xyxy, width, height, azimuth_deg, elevation_deg, fov_deg,
       background_kind, mask_coverage
FROM IDENTIFIER(:uc_schema || '.synthetic_views')
WHERE job_run_id = :job_run_id AND :refresh_bucket >= 0
ORDER BY view_id
LIMIT 48
