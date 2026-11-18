# SynthGen 3D

One photo of a manufacturing part becomes a trained object detector, end to end on Databricks:

1. **generate_3d_splat** – [IDOL](https://speridlabs.com/research/idol) reconstructs the photo as a 3D Gaussian splat (~262k Gaussians).
2. **render_synthetic_dataset** – [gsplat](https://github.com/nerfstudio-project/gsplat) renders a few hundred random views, composites them onto randomized backgrounds and derives pixel-exact bounding boxes from each render's alpha mask. Images and YOLO labels land in a Unity Catalog volume; per-view metadata lands in Delta.
3. **train_vision_detector** – fine-tunes an Ultralytics YOLO model on the synthetic set, tracks it in MLflow, registers it in Unity Catalog with a `@champion` alias, and runs it on the original photo as a sim→real check.

All three tasks run as one Lakeflow Job on serverless GPU. **SynthGen Studio**, an AppKit (React + TypeScript) Databricks App, starts runs, shows live progress, and has an interactive splat viewer, a dataset gallery with box overlays, and a model card.

In the app you can open three ready-made samples (traffic cone, tractor tire, ball bearing), start runs on your own photos, watch each run's progress, and cancel or delete your runs. Current status, known issues and the roadmap (next up: rare-obstacle detection for off-road autonomy, measured on real photos) are in [`docs/ROADMAP_V2.md`](docs/ROADMAP_V2.md).

```
Browser ──► SynthGen Studio (Databricks App, app service principal)
              ├─ upload photo ─────────────► UC volume  raw_inputs
              ├─ start run / poll status ──► Lakeflow Job (serverless GPU)
              │                                 T1 IDOL ─► splat_models/runs/<run_id>/model.ply
              │                                 T2 gsplat ─► synthetic_dataset/runs/<run_id>/ + synthetic_views (Delta)
              │                                 T3 YOLO ─► MLflow run + UC model <catalog>.synthgen_3d.synthgen_detector@champion
              ├─ splat viewer / gallery ◄── UC volumes (files plugin)
              └─ runs, metrics ◄────────── pipeline_runs / synthetic_views / training_epochs via SQL warehouse
```

## Repo layout

| Path | What it is |
|---|---|
| `databricks.yml`, `resources/jobs.yml` | Root bundle: the one-time setup job and the 3-task pipeline job |
| `src/00_setup.py` | Creates the schema, volumes and tables; builds the IDOL wheel; warms and caches Hugging Face weights and the gsplat CUDA build in a volume |
| `src/01_generate_splat.py`, `02_render_dataset.py`, `03_train_detector.py` | Pipeline tasks |
| `src/synthgen_common.py` | Shared helpers (progress files, Delta upserts, cache tarballs) |
| `spikes/` | The feasibility notebooks used to size GPUs and check the gsplat build; kept for reference |
| `app/synthgen-studio/` | The Databricks App (its own bundle) |

## Prerequisites

- A workspace with **serverless GPU compute** (the job defaults to `GPU_1xA10`; `GPU_1xH100` works if you have capacity) and a SQL warehouse.
- A Unity Catalog catalog you can create a schema in.
- A Hugging Face token with accepted terms for the gated repos IDOL pulls: `black-forest-labs/FLUX.2-dev` (VAE) and `facebook/dinov3-vith16plus`. Store it in a secret scope:
  ```bash
  databricks secrets create-scope synthgen
  databricks secrets put-secret synthgen huggingface
  ```
- A recent Databricks CLI (built with v1.17) and Node.js 22+.

## Deploy

**1. Pipeline bundle.** Set your values in `.databricks/bundle/dev/variable-overrides.json` (gitignored) or pass `--var` flags:

```json
{ "catalog": "my_catalog", "hf_secret_scope": "synthgen" }
```

```bash
databricks bundle deploy -p <profile>
databricks bundle run synthgen_setup -p <profile>   # one-time: schema, volumes, tables, model + CUDA caches
```

**2. Sample photos.** The app shows three curated samples (defined in `app/synthgen-studio/shared/samples.ts`). Download the photos from the links in the credits below and upload them to `/Volumes/<catalog>/synthgen_3d/raw_inputs/samples/` as `traffic_cone.jpg`, `tractor_tire.jpg` and `bearing_fag_6304.jpg`. The job's default parameter uses the bearing. Run the pipeline once per sample (job parameters `image_volume_path` and `target_class_name`) so each one opens instantly in the app.

**3. App bundle.** In `app/synthgen-studio/.databricks/bundle/default/variable-overrides.json`:

```json
{
  "sql_warehouse_id": "<warehouse id>",
  "pipeline_job_id": "<synthgen-3d-pipeline job id>",
  "catalog": "my_catalog",
  "sample_run_ids": ""
}
```

```bash
cd app/synthgen-studio
npm install
databricks bundle deploy -p <profile>
databricks bundle run app -p <profile>
```

The bundle grants the app's service principal the warehouse, job, volume and table permissions. It can't grant `USE CATALOG` / `USE SCHEMA`, so run that once:

```sql
GRANT USE CATALOG ON CATALOG my_catalog TO `<app service principal application id>`;
GRANT USE SCHEMA ON SCHEMA my_catalog.synthgen_3d TO `<app service principal application id>`;
```

Once the sample runs finish, set `sample_run_ids` to `traffic_cone:<run id>,tractor_tire:<run id>,bearing:<run id>` and redeploy. The samples then open on known-good runs, which helps when GPUs are slow to start during a demo. Any other runs show under "My objects".

## Local development (app)

Create `app/synthgen-studio/.env` (gitignored) with `DATABRICKS_CONFIG_PROFILE`, `DATABRICKS_WAREHOUSE_ID`, `DATABRICKS_JOB_PIPELINE` and the `DATABRICKS_VOLUME_*` paths, then `npm run dev`. The analytics queries bind `catalog.schema` at runtime with `IDENTIFIER()`. Generated query types are committed; if you change a query, run `npm run typegen` with the `-- @param uc_schema ... = <sample>` default pointing at a schema that exists in your workspace.

`three` and `@sparkjsdev/spark` are vendored under `app/synthgen-studio/vendor/` (MIT, licenses included) because the Databricks Apps npm mirror didn't serve `three` when this was built.

## Measured on GPU_1xA10 (300 views, 30 epochs)

| Step | Time |
|---|---|
| IDOL inference | ~65 s (5.4 GB VRAM) |
| generate_3d_splat task, incl. GPU start + cache restore | ~4.5 min |
| render_synthetic_dataset | ~8 min (most of it is JPEG encoding and volume writes) |
| train_vision_detector | ~7 min (YOLO training ~2 min) |

Validation metrics come from synthetic views, so they read high (mAP50 ≈ 0.99 on the sample bearing). The real-photo check on the model card is the more honest signal.

## Licensing notes

- **IDOL** code and weights are Apache-2.0, but the pipeline downloads gated third-party components: the FLUX.2-dev VAE (FLUX non-commercial license) and DINOv3 (DINOv3 license). Review those terms before using this beyond a demo.
- **Ultralytics YOLO** is AGPL-3.0 (enterprise licenses available from Ultralytics).
- **gsplat** Apache-2.0; **three.js** and **Spark** MIT.
- **Sample photos** (not included in this repo; the 3D models, renders and detectors made from them are derivatives under the same terms):
  - "[1990s FAG 6304 RSR single row deep groove ball bearing made in Germany, view 1](https://commons.wikimedia.org/wiki/File:1990s_FAG_6304_RSR_single_row_deep_groove_ball_bearing_made_in_Germany_view_1.jpg)" by R. Henrik Nilsson, CC BY 4.0.
  - "[Cône de signalisation](https://commons.wikimedia.org/wiki/File:C%C3%B4ne_de_signalisation.jpg)" by Filo gèn', CC BY-SA 4.0.
  - "[Tractor tyre on Metfield Common](https://commons.wikimedia.org/wiki/File:Tractor_tyre_on_Metfield_Common_-_geograph.org.uk_-_4214750.jpg)" by Evelyn Simak, CC BY-SA 2.0.

This is a demo, not a supported Databricks product.
