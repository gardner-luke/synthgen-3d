-- Per-epoch validation curve.
-- @param uc_schema STRING = main.synthgen_3d
-- @param job_run_id BIGINT = 0
-- @param refresh_bucket INT = 0
SELECT epoch, map50, map50_95, precision, recall, box_loss, cls_loss
FROM IDENTIFIER(:uc_schema || '.training_epochs')
WHERE job_run_id = :job_run_id AND :refresh_bucket >= 0
ORDER BY epoch
