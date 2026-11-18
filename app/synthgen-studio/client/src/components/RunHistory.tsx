import { useEffect, useMemo, useState } from 'react';
import { sql } from '@databricks/appkit-ui/js';
import { Skeleton, useAnalyticsQuery } from '@databricks/appkit-ui/react';
import { ChevronRight } from 'lucide-react';
import { StatusTag, runTone } from '@/components/StatusTag';
import { RunActions } from '@/components/RunActions';
import { activeProgress, type ActiveRun, type AppConfig } from '@/lib/api';

interface Props {
  ucSchema: string;
  bucket: number;
  selected: number | null;
  onSelect: (id: number) => void;
  samples: AppConfig['samples'];
  /** Start with "My objects" collapsed (e.g. demo mode) unless the user has toggled it before. */
  collapseMine?: boolean;
  /** In-flight runs from the Jobs API (they may not have a pipeline_runs row yet). */
  activeRuns: ActiveRun[];
  onChanged: (kind: 'canceled' | 'deleted', runId: number) => void;
}

interface MineItem {
  id: number;
  cls: string;
  time: string | number | null;
  status: string | null;
  live: ActiveRun | null;
}

/** Three thin segments: done steps solid, the running step pulsing. */
function StepBar({ done, running }: { done: number; running: number | null }) {
  return (
    <span className="mt-1 grid grid-cols-3 gap-0.5" aria-hidden>
      {[0, 1, 2].map((i) => (
        <span key={i} className={`h-1 rounded-sm ${i < done ? 'bg-primary' : i === running ? 'animate-pulse bg-primary/50' : 'bg-border'}`} />
      ))}
    </span>
  );
}

const MINE_KEY = 'synthgen.myObjectsOpen';

const fmtTime = (t: string | null | undefined) =>
  t ? new Date(t).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : '';

const rowCls = (active: boolean) =>
  `w-full rounded-md px-1.5 py-1.5 text-left transition-colors hover:bg-sidebar-accent disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:bg-transparent ${
    active ? 'bg-sidebar-accent text-sidebar-accent-foreground' : ''
  }`;

/** Sidebar, styled like the workspace left nav: curated samples pinned on top, then the user's own runs. */
export function RunHistory({ ucSchema, bucket, selected, onSelect, samples, collapseMine = false, activeRuns, onChanged }: Props) {
  const [mineOpen, setMineOpen] = useState(() => {
    const saved = window.localStorage.getItem(MINE_KEY);
    return saved === null ? !collapseMine : saved === '1';
  });
  useEffect(() => window.localStorage.setItem(MINE_KEY, mineOpen ? '1' : '0'), [mineOpen]);
  const params = useMemo(() => ({ uc_schema: sql.string(ucSchema), refresh_bucket: sql.number(bucket) }), [ucSchema, bucket]);
  const { data, loading, error } = useAnalyticsQuery('runs_recent', params, { autoStart: !!ucSchema });

  const sampleIds = useMemo(() => new Set(samples.map((s) => s.runId).filter((id): id is number => !!id)), [samples]);
  const byId = useMemo(() => new Map((data ?? []).map((r) => [Number(r.job_run_id), r])), [data]);
  const mine = useMemo<MineItem[]>(() => {
    const live = new Map(activeRuns.filter((r) => !sampleIds.has(r.runId)).map((r) => [r.runId, r]));
    const rows: MineItem[] = (data ?? [])
      .filter((r) => !sampleIds.has(Number(r.job_run_id)))
      .map((r) => ({ id: Number(r.job_run_id), cls: r.target_class_name ?? 'part', time: r.submitted_at, status: r.status, live: live.get(Number(r.job_run_id)) ?? null }));
    const known = new Set(rows.map((r) => r.id));
    const pending: MineItem[] = [...live.values()]
      .filter((r) => !known.has(r.runId))
      .map((r) => ({ id: r.runId, cls: r.className ?? 'part', time: r.startTime, status: null, live: r }));
    return [...pending, ...rows].sort((a, b) => Number(!!b.live) - Number(!!a.live));
  }, [data, activeRuns, sampleIds]);
  const runningCount = mine.filter((m) => m.live).length;

  return (
    <nav aria-label="Runs" className="flex min-h-0 flex-1 flex-col overflow-y-auto pb-3">
      <p className="px-3 pb-1 text-xs font-semibold text-muted-foreground">Samples</p>
      <div className="space-y-0.5 px-1.5">
        {samples.map((s) => {
          const r = s.runId ? byId.get(s.runId) : undefined;
          const { tone, label } = s.runId ? runTone(r?.status) : { tone: 'pending' as const, label: 'Not run yet' };
          const active = !!s.runId && selected === s.runId;
          return (
            <button
              key={s.key}
              type="button"
              data-demo={`sample-${s.key}`}
              disabled={!s.runId}
              onClick={() => s.runId && onSelect(s.runId)}
              aria-current={active ? 'page' : undefined}
              className={`${rowCls(active)} flex items-center gap-2.5`}
            >
              <img
                src={`/api/files/raw/raw?path=${encodeURIComponent(s.image)}`}
                alt=""
                className="h-9 w-9 shrink-0 rounded-sm border bg-background object-cover"
              />
              <span className="min-w-0 flex-1">
                <span className="block truncate font-semibold">{s.label}</span>
                <span className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                  <span className="truncate">{s.domain}</span>
                  {s.runId && loading ? <Skeleton className="h-3 w-12" /> : tone !== 'success' && <StatusTag tone={tone} label={label} small />}
                </span>
              </span>
            </button>
          );
        })}
      </div>

      <button
        type="button"
        onClick={() => setMineOpen((o) => !o)}
        aria-expanded={mineOpen}
        aria-controls="my-objects"
        className="mx-1.5 mt-4 flex items-center gap-1 rounded-sm px-1.5 pb-1 text-left text-xs font-semibold text-muted-foreground hover:text-foreground"
      >
        <ChevronRight className={`h-3.5 w-3.5 transition-transform ${mineOpen ? 'rotate-90' : ''}`} aria-hidden />
        My objects
        {!mineOpen && !!mine.length && (
          <span className="font-normal tabular-nums">{runningCount ? `(${runningCount} running)` : `(${mine.length})`}</span>
        )}
      </button>
      <div id="my-objects" hidden={!mineOpen} className="space-y-0.5 px-1.5">
        {loading && Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="mx-1.5 h-11" />)}
        {error && <p className="px-1.5 text-xs text-destructive">Could not load runs: {error}</p>}
        {!loading && !error && !mine.length && <p className="px-1.5 text-xs text-muted-foreground">No runs yet. Start one with New run.</p>}
        {mine.map((m) => {
          const active = selected === m.id;
          const prog = m.live ? activeProgress(m.live) : null;
          const { tone, label } = prog ? { tone: m.live?.queued ? ('queued' as const) : ('running' as const), label: prog.label } : runTone(m.status);
          return (
            <div key={m.id} className={`group relative rounded-md transition-colors hover:bg-sidebar-accent ${active ? 'bg-sidebar-accent' : ''}`}>
              <button
                type="button"
                onClick={() => onSelect(m.id)}
                aria-current={active ? 'page' : undefined}
                className={`w-full px-1.5 py-1.5 pr-8 text-left ${active ? 'text-sidebar-accent-foreground' : ''}`}
              >
                <span className="block truncate font-semibold">{m.cls}</span>
                <span className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                  <StatusTag tone={tone} label={label} small />
                  <span className="shrink-0 tabular-nums">
                    {m.live?.elapsedMs != null ? `${Math.max(1, Math.round(m.live.elapsedMs / 60000))}m` : fmtTime(m.time == null ? null : new Date(m.time).toISOString())}
                  </span>
                </span>
                {prog && <StepBar done={prog.done} running={prog.running} />}
              </button>
              <span className={`absolute right-1 top-1.5 ${active || m.live ? '' : 'opacity-0 group-hover:opacity-100 focus-within:opacity-100'}`}>
                <RunActions runId={m.id} label={m.cls} active={!!m.live} onChanged={onChanged} compact />
              </span>
            </div>
          );
        })}
      </div>
    </nav>
  );
}
