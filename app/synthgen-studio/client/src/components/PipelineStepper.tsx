import { useEffect, useState } from 'react';
import { Progress } from '@databricks/appkit-ui/react';
import { StatusTag, type Tone } from '@/components/StatusTag';
import { fmtDuration, type RunStatus, type StageProgress, type TaskState } from '@/lib/api';

const STEPS = [
  { key: 'generate_3d_splat', title: 'generate_3d_splat', sub: 'IDOL: photo to 3D Gaussian splat' },
  { key: 'render_synthetic_dataset', title: 'render_synthetic_dataset', sub: 'gsplat: rendered views with auto bounding boxes' },
  { key: 'train_vision_detector', title: 'train_vision_detector', sub: 'YOLO fine-tune, tracked in MLflow, registered in UC' },
] as const;

type Phase = 'pending' | 'queued' | 'starting' | 'running' | 'done' | 'failed' | 'skipped';

function phaseOf(t: TaskState | undefined): Phase {
  if (!t) return 'pending';
  const lc = t.lifeCycleState ?? '';
  if (lc === 'TERMINATED') return t.resultState === 'SUCCESS' ? 'done' : 'failed';
  if (lc === 'INTERNAL_ERROR') return 'failed';
  if (lc === 'SKIPPED') return 'skipped';
  if (lc === 'QUEUED') return 'queued';
  if (lc === 'PENDING') return 'starting';
  if (lc === 'RUNNING' || lc === 'TERMINATING') return 'running';
  return 'pending';
}

const PHASE_TAG: Record<Phase, { tone: Tone; label: string }> = {
  pending: { tone: 'pending', label: 'Waiting' },
  queued: { tone: 'queued', label: 'Queued' },
  starting: { tone: 'running', label: 'Starting GPU' },
  running: { tone: 'running', label: 'Running' },
  done: { tone: 'success', label: 'Succeeded' },
  failed: { tone: 'failed', label: 'Failed' },
  skipped: { tone: 'skipped', label: 'Skipped' },
};

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

interface Props {
  status: RunStatus | null;
  progress: Record<string, StageProgress | null>;
}

export function PipelineStepper({ status, progress }: Props) {
  const live = !!status && !['TERMINATED', 'INTERNAL_ERROR', 'SKIPPED'].includes(status.lifeCycleState ?? '');
  const now = useNow(live);
  return (
    <ol className="grid overflow-hidden rounded-lg border bg-card md:grid-cols-3 md:divide-x max-md:divide-y">
      {STEPS.map((s, i) => {
        const t = status?.tasks.find((x) => x.taskKey === s.key);
        const phase = status?.queued && i === 0 ? 'queued' : phaseOf(t);
        const p = progress[s.key] ?? null;
        const elapsed = t?.startTime ? (t.endTime || now) - t.startTime : null;
        const pct = p && p.total ? Math.min(100, Math.round(((p.done ?? 0) / p.total) * 100)) : phase === 'done' ? 100 : 0;
        const active = phase === 'running' || phase === 'starting';
        const tag = PHASE_TAG[phase];
        return (
          <li key={s.key} className={`relative p-4 ${active ? 'shadow-[inset_0_2px_0_var(--primary)]' : ''} ${phase === 'failed' ? 'shadow-[inset_0_2px_0_var(--destructive)]' : ''}`}>
            <div className="flex items-center justify-between gap-2">
              <StatusTag tone={tag.tone} label={tag.label} />
              <span className="text-xs tabular-nums text-muted-foreground">{fmtDuration(elapsed)}</span>
            </div>
            <p className="mt-2 truncate font-mono text-[13px] font-semibold">{s.title}</p>
            <p className="text-xs text-muted-foreground">{s.sub}</p>
            {(phase === 'running' || phase === 'done') && (
              <div className="mt-3 space-y-1">
                <Progress value={pct} className="h-1" />
                <p className="truncate text-xs text-muted-foreground">{p?.message ?? (phase === 'done' ? 'Complete' : 'Working…')}</p>
              </div>
            )}
            {phase === 'starting' && (
              <p className="mt-3 text-xs text-muted-foreground">Allocating serverless GPU{t?.setupMs ? ` · ${fmtDuration(t.setupMs)}` : ''}</p>
            )}
            {phase === 'failed' && t?.stateMessage && <p className="mt-3 line-clamp-3 text-xs text-destructive">{t.stateMessage}</p>}
          </li>
        );
      })}
    </ol>
  );
}
