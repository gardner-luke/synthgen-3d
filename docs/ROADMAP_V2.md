# SynthGen 3D v2 plan

## Direction (decided 2026-11-13): rare-obstacle detection for off-road autonomy
An autonomous tractor, sprayer or haul truck has to detect objects it almost never sees in training: a person or animal in the crop, a dropped tire, a cone, debris, an irrigation pipe. You can't stage enough of those in front of a moving machine to collect real data, and a miss is a safety incident. This pipeline handles that well: photograph the obstacle once, reconstruct it in 3D, and insert rendered views into real field imagery at realistic distances and angles.

- **Business impact:** better coverage of the long tail of rare, high-risk obstacles, and faster safety validation when moving into new crops, regions or job sites. The framing is generic off-road autonomy (ag, construction, mining), so it isn't tied to any one OEM's product.
- **Hero objects:** field obstacles. The traffic cone and tractor tire samples come first. Animals and equipment come later. Start without people, because of consent and how well a person reconstructs in 3D.
- **The bearing** stays as the manufacturing sample. It isn't the headline story.
- **Considered and set aside for now:** see-and-spray style weed detection (thin, dense plants are a poor fit for single-photo 3D, and real data is already plentiful there), defect detection (needs a defect-synthesis step we don't have), and parts identification (a good fallback with easy real data).

## The goal
Today the demo shows that the pipeline runs, but it doesn't show that the synthetic data helps. Validation mAP50-95 is about 0.995 because the validation images come from the same splat as the training images. A CV team will pick up on that quickly.

v2 needs one result a production team cares about: **"With N real labeled field images plus synthetic obstacles, recall on rare obstacles in real field photos went from X to Y, for $Z."** Everything in the first milestone serves that one sentence.

## What I'd change from the review feedback
- **Real labeled data is the critical dependency, so sort it out first.** The 3-way comparison (real only / synthetic only / real + synthetic) needs a small real *training* split as well as a real test set. I think the strongest story is the low-data case: about 30–50 real training images and about 100 real test images. That's a situation teams actually have, and it's where synthetic data tends to help most.
- **P3 as written would produce an all-green heatmap.** It slices predictions on synthetic views, and those are saturated. Failure analysis should start with real-image errors (a false positive / false negative gallery, sliced by box size, occlusion and lighting tags). Synthetic condition slicing only becomes useful once P4 makes the synthetic images harder.
- **Move part of P7 earlier.** A live run that fails 4 times before training succeeds is a bigger demo risk than missing features. The root-cause fix and the cheap speedups go in milestone 1.
- **Defer P2, P6 serving, and P8.** "Bring your model" and closed-loop serving are probably the right long-term story, but each one is a project on its own. Do them once the real-image numbers hold up.
- **Expect a sim-to-real gap.** The splat stores flat color only (SH degree 0), and the inserted obstacles won't pick up field lighting, shadows or mud at first. Real field photos will probably expose that, which gives M2 something measurable to fix.

## Milestones

### M1: credible accuracy (P1, P7 fixes, registry hygiene)
**Data**
- New volume `real_datasets/<eval_set_id>/` in YOLO layout (images/, labels/, data.yaml), with train/test splits.
- New table `real_eval_sets` with these columns: eval_set_id, class_name, n_train, n_test, source, license, created_by, created_at, sha256 manifest.
- New table `eval_results` with these columns: job_run_id, eval_set_id, model_variant (`real_only` | `synthetic_only` | `real_plus_synthetic`), split, metric, value, mlflow_run_id.
- New table `predictions` with these columns: job_run_id, model_variant, image_path, pred_bbox, conf, iou, outcome (TP/FP/FN). Covers real test images only for now.
- No changes to existing columns. `pipeline_runs` gets a nullable `eval_set_id` and a `real_map50_95` headline column, so the run list can sort on it.

**Pipeline**
- The T3 job parameter `eval_set_id` is optional. If it's empty, today's behavior is unchanged.
- When it's set, T3 trains 3 variants as child MLflow runs under one parent, using the same epochs/imgsz/seed. Then it evaluates all 3 on the real test split and writes `eval_results` and `predictions`.
  - Running them one after another on one A10 should add about 4–6 min, since YOLO-n training is about 2 min. If that's too slow live, run the 3 variants as parallel tasks.
- Registry:
  - One UC model per class, named `<catalog>.synthgen_3d.detector_<class>`.
  - Each new version gets the `@challenger` alias.
  - The `@champion` alias moves only when real mAP50-95 beats the current champion.
- Reliability:
  - Pull the logs for the 4 failed T3 attempts and fix the root cause.
  - Fix the duplicate epoch-31 row in `training_epochs` (probably the final-validation callback writing twice).
  - Add task `max_retries: 1` and a clear error message in `pipeline_runs`.
- `submitted_by`: when a run is started from the Jobs UI or CLI, there's no app header, so T1 should fall back to the run's `creator_user_name`.

**App**
- Add an "Evaluate on real photos" option to the New run dialog. It can upload a zipped YOLO/COCO set or pick an existing set.
- Add a new **Evaluation** tab, which becomes the default once results exist:
  - real-image mAP50 / mAP50-95 / recall for the 3 variants
  - the synthetic-vs-real gap shown next to them
  - a gallery of real test images with TP/FP/FN boxes
- The model card shows synthetic metrics labeled as "synthetic validation". The headline number becomes real-image mAP.

**Done when:** the cone and tire runs show all 3 variants on a real test set in the app, and the champion alias is gated on real mAP.

### M2: realistic synthetic data (P4)
- Composite onto real background photos: a small CC0/CC-BY set in a `backgrounds` volume, plus optional user uploads.
- Several instances per image with occlusion, distractor objects, and 10–20% negative images with no target.
- Photometric and sensor randomization: color temperature, exposure, noise, motion/defocus blur, JPEG quality.
- Target camera intrinsics and non-square resolution.
- Log every randomization parameter per image as new nullable columns in `synthetic_views`.
- Stretch: export SH degree > 0 from IDOL, if the model supports it, for view-dependent shine.

**Done when:** real-image mAP on the same eval set beats the M1 synthetic-only and real + synthetic numbers, and each randomization is logged per image.

### M3: find and fix weak spots (P3)
- Slice real-image errors by box size, truncation, and lighting/background tags. The tags can come from a quick `ai_query` VLM pass over the test set.
- Once M2 lands, slice synthetic predictions by azimuth/elevation/radius/background/randomization parameters, and show them as a heatmap.
- Add a "Generate more here" action that re-runs T2+T3 with camera/randomization sampling weighted toward the weak bins.
- Show cost per run from `system.billing.usage` joined on `usage_metadata.job_run_id`. Billing rows can lag by hours, so show an estimate first and fill in the actual later.

### M4: production lifecycle (P2, P6)
- "Bring your model + dataset": baseline eval of an existing UC model, targeted generation, and per-class recall before and after.
- Serving endpoint with inference tables, following the Model Workbench pattern. Low-confidence traffic goes into a labeling and generation queue.

### Later
- P5: COCO export, instance masks (from the alpha we already render), 6-DoF pose from world_to_camera, dataset versioning and a dataset card.
- P8: multi-photo, turntable video, or CAD/mesh inputs.
- Speed:
  - a custom container image (Public Preview) or a wheelhouse in a volume to cut the roughly 133 s of pip installs
  - merge T1+T2 to save a cold start
  - threaded JPEG encoding plus a single tar write in T2

  These should bring a run to roughly 12–15 min. Getting under 10 min probably needs multi-GPU rendering.

## Status checkpoint (2026-11-17)

### Working today (deployed)
- **Pipeline:** one photo → IDOL 3D Gaussian splat → 300 gsplat renders with alpha-mask boxes → YOLO fine-tune → MLflow + UC model `@champion`. About 18–20 min per run on `GPU_1xA10`, roughly $1–2.
- **Samples:** three curated samples, each with a finished run, pinned in the sidebar: traffic cone and tractor tire (field obstacles) and the ball bearing (manufacturing). They're defined in `shared/samples.ts` and mapped to runs through the `sample_run_ids` bundle variable.
- **My objects:**
  - Ad-hoc runs. The section is collapsible, and starts collapsed with `?demo=1`.
  - In-flight runs appear within seconds, showing the step (n/3), a progress bar and elapsed minutes.
  - Each run has a "…" menu: **Cancel** (while running) and **Delete** (finished or cancelled).
  - Delete removes the run's table rows, splat and dataset files, and the job-run record. It keeps the UC model version. Sample runs are protected.
- **New run dialog:** pick a sample or upload a photo, with Quick/Full presets.
- **Viewer:**
  - Splats are converted from IDOL's OpenCV convention so they load upright.
  - `/api/runs/:id/splat` caches splats in memory and sets an immutable browser cache, so repeat loads take under a second. Sample splats are warmed at startup.
- **Demo mode:** `?demo=1` hides the UC path and submitter email.
- **Databricks look:** Du Bois tokens, flat cards, underline tabs, icon+text status tags.

### Known issues and tech debt
- **Typegen clobber.** `npm run dev` (the AppKit Vite plugin) regenerates `shared/appkit-types/analytics.d.ts`. The query sample default `main.synthgen_3d` usually doesn't exist, so the file comes out with `result: unknown`.
  - Workaround: regenerate with the sample default pointed at a real schema, or restore the file before committing.
  - Real fix to look into: a typegen-only schema override, or a committed types file the dev plugin doesn't overwrite.
- **Validation metrics are saturated** (mAP50-95 ≈ 0.995 on synthetic val). This is the main credibility gap, and M1 addresses it.
- **Cold splat loads** through the files API take 30–75 s. They're cached after the first load, but the first view of a new run is still slow.
- **Camera sampling** covers the full sphere, including from underneath. For field obstacles, M2 should bias toward near-ground elevations and realistic distances.
- **Duplicate epoch-31 row** in `training_epochs` (probably the final-validation callback writing twice).
- **Root cause of the earlier repeated `train_vision_detector` attempts** isn't confirmed. Recent runs have all succeeded first try.
- **Deleting a run keeps its UC model version.** Classes are all versions of one model name today, so per-class model names (M1) would make cleanup safer.
- **`submitted_by` is empty** for runs started outside the app (Jobs UI/CLI). T1 should fall back to the run's creator.

### Parked: demo recording and a reusable skill
- testreel scripts are drafted in `demo/` (`preview.yaml`, `cone.yaml`). Record against the local dev server with `?demo=1`.
- **Gotcha:** testreel needs the ffmpeg bundled with `ffmpeg-static` (6.x). Homebrew ffmpeg 9 removed `-filter_complex_script`, which testreel uses. Use `--headed` so WebGL uses the GPU.
- Outputs to target: MP4 for LinkedIn and Slack, a gifski GIF under about 8 MB for the README.
- Later, turn the workflow into a personal `demo-reel` skill:
  - describe the flows in plain English
  - the agent finds selectors, drafts a storyboard and records
  - it reviews extracted frames, then exports per target
  - privacy checklist before anything is shared

### Other ideas captured
- **Docs:** ARCHITECTURE (diagram), DEMO_RUNBOOK, PERF_CHECKLIST with measured numbers, LICENSING (IDOL's gated FLUX.2-dev / DINOv3, Ultralytics AGPL, CC BY-SA sample photos).
- **Hero image:** a 3/4-angle shot for the README and social posts.
- **Serving:** an optional deploy step for a serving endpoint and an endpoint badge in the app.
- **Cost:** per-run cost from `system.billing.usage` (by `job_run_id`) on the model card.
- **License:** the repo has no LICENSE file yet. Choose one, for example Apache-2.0 or MIT.

## Decisions needed
1. **Where the real labeled field images come from.** M1 needs real field photos with obstacles in them, both for a small real training split and for the test set. The options are a public off-road or agricultural obstacle dataset (check license and how well its classes match), staging and photographing cones and tires in a field ourselves, or a mix of the two. Field backgrounds without obstacles are easier to get, and M2 needs them for compositing.
2. **Obstacle classes for M1.** Start with cone + tire, both with samples already, or add an animal class right away.
3. **M1 scope.** Is M1 alone (about 1–1.5 weeks) the next demo, or does the next customer showing need M2 as well?

## Constraints carried over
- Keep provenance end to end: source_sha256 → dataset → MLflow parent and child runs → UC model version, plus eval_set_id and its manifest hash.
- Only add tables or nullable columns, so the current app keeps working against old runs.
- Keep lab identifiers out of the public repo, with values in variable-overrides as today.
