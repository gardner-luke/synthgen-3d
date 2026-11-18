import { createApp, analytics, files, jobs, server, getExecutionContext } from '@databricks/appkit';
import { z } from 'zod';
import { SAMPLES } from '../shared/samples';

/**
 * SynthGen Studio backend.
 *
 * Plugins do the heavy lifting:
 *  - files:     uploads to raw_inputs, streams .ply / renders from splat_models + synthetic_dataset
 *  - jobs:      triggers the Lakeflow pipeline (job_parameters pass straight through when taskType is unset)
 *  - analytics: config/queries/*.sql against the SQL warehouse
 *
 * Custom routes cover what the plugins don't: an *uncached* run-status read (the jobs plugin caches
 * reads for 60s, too stale for a live stepper) and a small config endpoint for deep links.
 */

const env = (k: string, d = '') => process.env[k] ?? d;
const VOLUMES = {
  raw: env('DATABRICKS_VOLUME_RAW'),
  splats: env('DATABRICKS_VOLUME_SPLATS'),
  dataset: env('DATABRICKS_VOLUME_DATASET'),
};

// "catalog.schema" for the analytics queries, derived from the raw_inputs volume path (/Volumes/<catalog>/<schema>/<volume>).
const UC_SCHEMA = VOLUMES.raw.split('/').slice(2, 4).join('.');

// SAMPLE_RUN_IDS="traffic_cone:123,bearing:456" maps each curated sample to its finished run in this workspace.
// GOLDEN_RUN_ID is the older single-sample setting and still fills in the bearing if it isn't mapped.
const sampleRunIds = new Map(
  env('SAMPLE_RUN_IDS')
    .split(',')
    .map((pair) => pair.split(':').map((x) => x.trim()))
    .filter(([k, v]) => k && Number(v) > 0)
    .map(([k, v]) => [k, Number(v)] as const),
);
if (!sampleRunIds.has('bearing') && Number(env('GOLDEN_RUN_ID')) > 0) sampleRunIds.set('bearing', Number(env('GOLDEN_RUN_ID')));
const SAMPLE_CONFIG = SAMPLES.map((s) => ({ ...s, runId: sampleRunIds.get(s.key) ?? null }));

// Splats are ~18 MB and never change once written, but streaming one through the files API takes 30-75 s.
// Keep the most recent few in memory (sample runs are warmed at startup) and let the browser cache them.
const SPLAT_CACHE_MAX = 6;
const splatCache = new Map<number, Promise<Buffer>>();
function loadSplat(runId: number): Promise<Buffer> {
  const hit = splatCache.get(runId);
  if (hit) {
    splatCache.delete(runId);
    splatCache.set(runId, hit); // refresh LRU position
    return hit;
  }
  const p = (async () => {
    const { contents } = await getExecutionContext().client.files.download({ file_path: `${VOLUMES.splats}/runs/${runId}/model.ply` });
    if (!contents) throw new Error('empty splat download');
    return Buffer.from(await new Response(contents).arrayBuffer());
  })();
  p.catch(() => splatCache.delete(runId)); // don't cache failures (e.g. the splat isn't written yet)
  splatCache.set(runId, p);
  while (splatCache.size > SPLAT_CACHE_MAX) splatCache.delete(splatCache.keys().next().value!);
  return p;
}

const PRESETS = {
  quick: { gaussian_count: 131072, idol_steps: 30, num_views: 120, epochs: 12, image_size: 512 },
  full: { gaussian_count: 262144, idol_steps: 50, num_views: 300, epochs: 30, image_size: 640 },
} as const;

const runRequest = z.object({
  imagePath: z
    .string()
    .min(1)
    .max(512)
    .refine((p) => !p.includes('..') && !p.startsWith('/'), 'imagePath must be relative to the raw_inputs volume'),
  targetClassName: z
    .string()
    .min(1)
    .max(40)
    .regex(/^[A-Za-z0-9 _-]+$/, 'letters, numbers, spaces, - and _ only'),
  preset: z.enum(['quick', 'full']).default('full'),
});

interface TaskState {
  taskKey: string;
  lifeCycleState: string | null;
  resultState: string | null;
  stateMessage: string | null;
  startTime: number | null;
  endTime: number | null;
  setupMs: number | null;
}

const TASK_ORDER = ['generate_3d_splat', 'render_synthetic_dataset', 'train_vision_detector'];

// ---- Run management (cancel / delete) --------------------------------------------------------------
const TABLE = (name: string) => {
  if (!/^[\w-]+\.[\w-]+$/.test(UC_SCHEMA)) throw new Error('UC schema not configured');
  const [cat, sch] = UC_SCHEMA.split('.');
  return `\`${cat}\`.\`${sch}\`.\`${name}\``;
};

async function runSql(statement: string, params: Record<string, string>) {
  const r = await getExecutionContext().client.statementExecution.executeStatement({
    warehouse_id: env('DATABRICKS_WAREHOUSE_ID'),
    statement,
    parameters: Object.entries(params).map(([name, value]) => ({ name, value })),
    wait_timeout: '50s',
  });
  if (r.status?.state !== 'SUCCEEDED') throw new Error(r.status?.error?.message ?? `statement ${r.status?.state}`);
}

/** Recursively delete a volume directory (the files API only removes empty directories). */
async function deleteVolumeDir(dir: string) {
  const files = getExecutionContext().client.files;
  const dirs: string[] = [];
  const walk = async (d: string) => {
    dirs.push(d);
    const paths: string[] = [];
    try {
      for await (const e of files.listDirectoryContents({ directory_path: d })) {
        if (!e.path) continue;
        if (e.is_directory) await walk(e.path);
        else paths.push(e.path);
      }
    } catch {
      return; // already gone
    }
    for (let i = 0; i < paths.length; i += 16) await Promise.all(paths.slice(i, i + 16).map((file_path) => files.delete({ file_path }).catch(() => undefined)));
  };
  await walk(dir);
  for (const d of dirs.reverse()) await files.deleteDirectory({ directory_path: d }).catch(() => undefined);
}

const SAMPLE_RUN_IDS = new Set(SAMPLE_CONFIG.map((x) => x.runId).filter((x): x is number => !!x));
const jobParam = (run: { job_parameters?: { name?: string; value?: string; default?: string }[] }, name: string) => {
  const p = run.job_parameters?.find((x) => x.name === name);
  return p?.value ?? p?.default ?? null;
};

createApp({
  plugins: [
    analytics(),
    files({
      volumes: {
        raw: { policy: files.policy.allowAll(), maxUploadSize: 25_000_000 },
        splats: { policy: files.policy.publicRead(), customContentTypes: { '.ply': 'application/octet-stream' } },
        dataset: { policy: files.policy.publicRead() },
        // Required by AppKit's static manifest (see DATABRICKS_VOLUME_FILES in app.yaml). Not used by the UI.
        files: { policy: files.policy.denyAll() },
      },
    }),
    jobs(),
    server(),
  ],
  onPluginsReady(appkit) {
    for (const s of SAMPLE_CONFIG) if (s.runId) loadSplat(s.runId).catch((e: unknown) => console.warn(`splat warm-up failed for ${s.key}:`, e));
    appkit.server.extend((app) => {
      app.get('/api/config', (_req, res) => {
        const host = getExecutionContext().client.config.host ?? env('DATABRICKS_HOST');
        res.json({
          host: host ? (host.startsWith('http') ? host : `https://${host}`) : '',
          jobId: Number(env('DATABRICKS_JOB_PIPELINE')) || null,
          volumes: VOLUMES,
          ucSchema: UC_SCHEMA,
          samples: SAMPLE_CONFIG,
          goldenRunId: SAMPLE_CONFIG.find((x) => x.runId)?.runId ?? null,
          presets: PRESETS,
        });
      });

      app.post('/api/runs', async (req, res) => {
        const parsed = runRequest.safeParse(req.body);
        if (!parsed.success) {
          res.status(400).json({ error: parsed.error.issues.map((i) => i.message).join('; ') });
          return;
        }
        const { imagePath, targetClassName, preset } = parsed.data;
        const p = PRESETS[preset];
        const submittedBy = String(req.header('x-forwarded-email') ?? req.header('x-forwarded-user') ?? 'local-dev');
        const job_parameters: Record<string, string> = {
          image_volume_path: `${VOLUMES.raw}/${imagePath}`,
          target_class_name: targetClassName.trim().toLowerCase().replace(/\s+/g, '_'),
          gaussian_count: String(p.gaussian_count),
          idol_steps: String(p.idol_steps),
          num_views: String(p.num_views),
          epochs: String(p.epochs),
          image_size: String(p.image_size),
          submitted_by: submittedBy,
        };
        const result = await appkit.jobs('pipeline').runNow({ job_parameters });
        if (!result.ok) {
          res.status(502).json({ error: 'Could not start the pipeline job' });
          return;
        }
        res.status(201).json({ runId: result.data.run_id, preset, params: job_parameters });
      });

      // Live list of this app's in-flight runs. Runs only get a pipeline_runs row once the GPU task starts,
      // so the sidebar merges these in to show queued/starting runs and per-task progress.
      app.get('/api/runs/active', async (_req, res) => {
        const jobId = Number(env('DATABRICKS_JOB_PIPELINE'));
        if (!jobId) {
          res.json({ runs: [] });
          return;
        }
        try {
          const runs = [];
          for await (const r of getExecutionContext().client.jobs.listRuns({ job_id: jobId, active_only: true, expand_tasks: true, limit: 25 })) {
            const latest = new Map<string, NonNullable<typeof r.tasks>[number]>();
            for (const t of r.tasks ?? []) {
              const prev = latest.get(t.task_key ?? '');
              if (!prev || (t.attempt_number ?? 0) >= (prev.attempt_number ?? 0)) latest.set(t.task_key ?? '', t);
            }
            runs.push({
              runId: r.run_id,
              className: jobParam(r, 'target_class_name'),
              submittedBy: jobParam(r, 'submitted_by'),
              startTime: r.start_time || null,
              elapsedMs: r.start_time ? Date.now() - r.start_time : null,
              lifeCycleState: r.state?.life_cycle_state ?? null,
              queued: r.state?.life_cycle_state === 'QUEUED',
              tasks: TASK_ORDER.map((k) => {
                const t = latest.get(k);
                return { taskKey: k, lifeCycleState: t?.state?.life_cycle_state ?? null, resultState: t?.state?.result_state ?? null, startTime: t?.start_time || null };
              }),
            });
          }
          res.set('Cache-Control', 'no-store').json({ runs });
        } catch (e) {
          res.status(502).json({ error: (e instanceof Error ? e.message : String(e)).slice(0, 300) });
        }
      });

      const ownRun = async (raw: string) => {
        const runId = Number(raw);
        if (!Number.isSafeInteger(runId) || runId <= 0) return { ok: false, error: 'invalid run id', code: 400 } as const;
        const run = await getExecutionContext().client.jobs.getRun({ run_id: runId });
        const jobId = Number(env('DATABRICKS_JOB_PIPELINE'));
        if (jobId && run.job_id !== jobId) return { ok: false, error: 'run not found for this app', code: 404 } as const;
        return { ok: true, runId, run } as const;
      };

      app.post('/api/runs/:runId/cancel', async (req, res) => {
        try {
          const own = await ownRun(req.params.runId);
          if (!own.ok) {
            res.status(own.code).json({ error: own.error });
            return;
          }
          const { runId, run } = own;
          await getExecutionContext().client.jobs.cancelRun({ run_id: runId });
          // Record the cancel so the run list doesn't show it as in progress forever (the row may not exist yet).
          await runSql(
            `MERGE INTO ${TABLE('pipeline_runs')} t USING (SELECT CAST(:id AS BIGINT) AS job_run_id) s ON t.job_run_id = s.job_run_id
             WHEN MATCHED THEN UPDATE SET status = 'canceled', updated_at = current_timestamp()
             WHEN NOT MATCHED THEN INSERT (job_run_id, submitted_by, submitted_at, target_class_name, status, updated_at)
               VALUES (s.job_run_id, :by, timestamp_millis(CAST(:at AS BIGINT)), :cls, 'canceled', current_timestamp())`,
            { id: String(runId), by: jobParam(run, 'submitted_by') ?? '', at: String(run.start_time ?? Date.now()), cls: jobParam(run, 'target_class_name') ?? 'part' },
          );
          res.json({ ok: true });
        } catch (e) {
          res.status(502).json({ error: (e instanceof Error ? e.message : String(e)).slice(0, 300) });
        }
      });

      app.delete('/api/runs/:runId', async (req, res) => {
        try {
          const own = await ownRun(req.params.runId);
          if (!own.ok) {
            res.status(own.code).json({ error: own.error });
            return;
          }
          const { runId, run } = own;
          if (SAMPLE_RUN_IDS.has(runId)) {
            res.status(403).json({ error: 'Sample runs can not be deleted' });
            return;
          }
          if (!['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR'].includes(run.state?.life_cycle_state ?? '')) {
            res.status(409).json({ error: 'Cancel the run before deleting it' });
            return;
          }
          for (const t of ['training_epochs', 'synthetic_views', 'pipeline_runs']) await runSql(`DELETE FROM ${TABLE(t)} WHERE job_run_id = CAST(:id AS BIGINT)`, { id: String(runId) });
          splatCache.delete(runId);
          // Files and the job-run record go in the background: the dataset folder holds hundreds of files.
          void (async () => {
            await Promise.all([deleteVolumeDir(`${VOLUMES.splats}/runs/${runId}`), deleteVolumeDir(`${VOLUMES.dataset}/runs/${runId}`)]);
            await getExecutionContext().client.jobs.deleteRun({ run_id: runId }).catch(() => undefined);
          })().catch((e: unknown) => console.warn(`cleanup for run ${runId} failed:`, e));
          res.json({ ok: true });
        } catch (e) {
          res.status(502).json({ error: (e instanceof Error ? e.message : String(e)).slice(0, 300) });
        }
      });

      app.get('/api/runs/:runId/splat', async (req, res) => {
        const runId = Number(req.params.runId);
        if (!Number.isSafeInteger(runId) || runId <= 0) {
          res.status(400).json({ error: 'invalid run id' });
          return;
        }
        try {
          const buf = await loadSplat(runId);
          res.set({ 'Content-Type': 'application/octet-stream', 'Content-Length': String(buf.length), 'Cache-Control': 'private, max-age=604800, immutable' });
          res.end(buf);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          res.status(404).json({ error: msg.slice(0, 300) });
        }
      });

      app.get('/api/runs/:runId/status', async (req, res) => {
        const runId = Number(req.params.runId);
        if (!Number.isSafeInteger(runId) || runId <= 0) {
          res.status(400).json({ error: 'invalid run id' });
          return;
        }
        try {
          const run = await getExecutionContext().client.jobs.getRun({ run_id: runId });
          const jobId = Number(env('DATABRICKS_JOB_PIPELINE'));
          if (jobId && run.job_id !== jobId) {
            res.status(404).json({ error: 'run not found for this app' });
            return;
          }
          // After a repair (or task retries) getRun lists every attempt of a task. Keep the latest one:
          // an attempt still in flight (no end_time) beats finished ones, then the newest start wins.
          const latest = new Map<string, NonNullable<typeof run.tasks>[number]>();
          for (const t of run.tasks ?? []) {
            const key = t.task_key ?? '';
            const prev = latest.get(key);
            const rank = (x: typeof t) => (x.end_time ? 0 : 1e15) + (x.start_time || 0) + (x.attempt_number ?? 0);
            if (!prev || rank(t) >= rank(prev)) latest.set(key, t);
          }
          const tasks: TaskState[] = [...latest.values()]
            .map((t) => ({
              taskKey: t.task_key ?? '',
              lifeCycleState: t.state?.life_cycle_state ?? null,
              resultState: t.state?.result_state ?? null,
              stateMessage: t.state?.state_message ?? null,
              startTime: t.start_time || null,
              endTime: t.end_time || null,
              setupMs: t.setup_duration ?? null,
            }))
            .sort((a, b) => TASK_ORDER.indexOf(a.taskKey) - TASK_ORDER.indexOf(b.taskKey));
          res.set('Cache-Control', 'no-store').json({
            runId,
            lifeCycleState: run.state?.life_cycle_state ?? null,
            resultState: run.state?.result_state ?? null,
            stateMessage: run.state?.state_message ?? null,
            queued: run.state?.life_cycle_state === 'QUEUED',
            startTime: run.start_time || null,
            endTime: run.end_time || null,
            runPageUrl: run.run_page_url ?? null,
            tasks,
          });
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          res.status(502).json({ error: msg.slice(0, 300) });
        }
      });
    });
  },
}).catch(console.error);
