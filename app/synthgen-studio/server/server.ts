import { createApp, analytics, files, jobs, server, getExecutionContext } from '@databricks/appkit';
import { z } from 'zod';

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
    appkit.server.extend((app) => {
      app.get('/api/config', (_req, res) => {
        const host = getExecutionContext().client.config.host ?? env('DATABRICKS_HOST');
        res.json({
          host: host ? (host.startsWith('http') ? host : `https://${host}`) : '',
          jobId: Number(env('DATABRICKS_JOB_PIPELINE')) || null,
          volumes: VOLUMES,
          ucSchema: UC_SCHEMA,
          goldenRunId: Number(env('GOLDEN_RUN_ID')) || null,
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
