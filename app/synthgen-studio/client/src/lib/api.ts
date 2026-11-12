import { useEffect, useMemo, useRef, useState } from 'react';

export type VolumeKey = 'raw' | 'splats' | 'dataset';

export interface AppConfig {
  host: string;
  jobId: number | null;
  volumes: Record<VolumeKey, string>;
  /** catalog.schema the pipeline writes to; bound into every analytics query. */
  ucSchema: string;
  goldenRunId: number | null;
  presets: Record<'quick' | 'full', { gaussian_count: number; idol_steps: number; num_views: number; epochs: number; image_size: number }>;
}

export interface TaskState {
  taskKey: string;
  lifeCycleState: string | null;
  resultState: string | null;
  stateMessage: string | null;
  startTime: number | null;
  endTime: number | null;
  setupMs: number | null;
}

export interface RunStatus {
  runId: number;
  lifeCycleState: string | null;
  resultState: string | null;
  stateMessage: string | null;
  queued: boolean;
  startTime: number | null;
  endTime: number | null;
  runPageUrl: string | null;
  tasks: TaskState[];
}

export interface EpochPoint {
  epoch: number;
  map50: number;
  map50_95: number;
  precision: number;
  recall: number;
}

export interface StageProgress {
  stage: string;
  done: number | null;
  total: number | null;
  message: string | null;
  elapsed_sec: number;
  updated_epoch: number;
  history?: EpochPoint[];
  [k: string]: unknown;
}

/** Pull `{ error }` out of a JSON error body without trusting its shape. */
export async function readError(r: Response, fallback: string): Promise<string> {
  try {
    const j: unknown = await r.json();
    if (j && typeof j === 'object' && 'error' in j) {
      const e = (j as { error: unknown }).error;
      if (typeof e === 'string') return e;
    }
  } catch {
    /* non-JSON body */
  }
  return fallback;
}

const TERMINAL = new Set(['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR']);
export const isTerminal = (s: RunStatus | null) => !!s && TERMINAL.has(s.lifeCycleState ?? '');

/** Turn an absolute /Volumes path from Delta into a Files-plugin URL for the matching volume key. */
export function volumeUrl(cfg: AppConfig | null, key: VolumeKey, absPath: string | null | undefined, action: 'raw' | 'download' = 'raw') {
  if (!cfg || !absPath) return null;
  const root = cfg.volumes[key];
  if (!root || !absPath.startsWith(root + '/')) return null;
  const rel = absPath.slice(root.length + 1);
  return `/api/files/${key}/${action}?path=${encodeURIComponent(rel)}`;
}

export function useConfig() {
  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetch('/api/config')
      .then(async (r) => {
        if (!r.ok) throw new Error(`config ${r.status}`);
        setCfg((await r.json()) as AppConfig);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);
  return { cfg, error };
}

/** Polls the uncached run-status route until the run reaches a terminal state. */
export function useRunStatus(runId: number | null, intervalMs = 4000) {
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!runId) {
      setStatus(null);
      return;
    }
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const r = await fetch(`/api/runs/${runId}/status`, { cache: 'no-store' });
        if (!r.ok) throw new Error(await readError(r, `status ${r.status}`));
        const s = (await r.json()) as RunStatus;
        if (stop) return;
        setStatus(s);
        setError(null);
        // Terminal runs keep a slow poll so a repair (re-running a failed task) shows up live.
        timer = setTimeout(() => void tick(), TERMINAL.has(s.lifeCycleState ?? '') ? intervalMs * 8 : intervalMs);
      } catch (e) {
        if (stop) return;
        setError(e instanceof Error ? e.message : String(e));
        timer = setTimeout(() => void tick(), intervalMs * 2);
      }
    };
    setStatus(null);
    void tick();
    return () => {
      stop = true;
      if (timer) clearTimeout(timer);
    };
  }, [runId, intervalMs]);
  return { status, error };
}

/** Reads a task's small progress JSON from its run folder (Files /raw is uncached). */
export function useStageProgress(runId: number | null, key: VolumeKey, stage: string, active: boolean, intervalMs = 4000) {
  const [state, setState] = useState<{ runId: number; data: StageProgress } | null>(null);
  const activeRef = useRef(active);
  useEffect(() => {
    activeRef.current = active;
  }, [active]);
  useEffect(() => {
    if (!runId) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const r = await fetch(`/api/files/${key}/raw?path=${encodeURIComponent(`runs/${runId}/progress_${stage}.json`)}&t=${Date.now()}`, {
          cache: 'no-store',
        });
        if (r.ok && !stop) setState({ runId, data: (await r.json()) as StageProgress });
      } catch {
        /* progress is best-effort */
      }
      if (!stop && activeRef.current) timer = setTimeout(() => void tick(), intervalMs);
    };
    void tick();
    return () => {
      stop = true;
      if (timer) clearTimeout(timer);
    };
  }, [runId, key, stage, active, intervalMs]);
  return state && state.runId === runId ? state.data : null;
}

/** A minute-granularity bucket that busts the analytics query cache while a run is live. */
export function useRefreshBucket(live: boolean) {
  const [bucket, setBucket] = useState(() => Math.floor(Date.now() / 30000));
  useEffect(() => {
    if (!live) return;
    const id = setInterval(() => setBucket(Math.floor(Date.now() / 30000)), 30000);
    return () => clearInterval(id);
  }, [live]);
  return bucket;
}

export function fmtDuration(ms: number | null | undefined) {
  if (ms == null || ms < 0) return '—';
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
}

export const fmtPct = (v: number | string | null | undefined, digits = 1) =>
  v == null || v === '' || Number.isNaN(Number(v)) ? '—' : `${(Number(v) * 100).toFixed(digits)}%`;

/** IDOL writes its predicted input camera next to the splat; use it to open the viewer on the photo's viewpoint. */
export function useInitialView(runId: number | null) {
  const [view, setView] = useState<{ runId: number; yaw: number; pitch: number } | null>(null);
  const [settled, setSettled] = useState<number | null>(null);
  useEffect(() => {
    if (!runId) return;
    let stop = false;
    fetch(`/api/files/splats/raw?path=${encodeURIComponent(`runs/${runId}/camera.json`)}`)
      .then(async (r) => {
        if (!r.ok) return;
        const j: unknown = await r.json();
        if (stop || !j || typeof j !== 'object') return;
        const { yaw, pitch } = j as { yaw?: unknown; pitch?: unknown };
        if (typeof yaw === 'number' && typeof pitch === 'number') setView({ runId, yaw, pitch });
      })
      .catch(() => undefined)
      .finally(() => {
        if (!stop) setSettled(runId);
      });
    return () => {
      stop = true;
    };
  }, [runId]);
  const current = useMemo(() => (view && view.runId === runId ? { yaw: view.yaw, pitch: view.pitch } : null), [view, runId]);
  return { view: current, ready: settled === runId };
}
