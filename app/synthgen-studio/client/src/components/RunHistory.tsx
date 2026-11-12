import { useMemo } from 'react';
import { sql } from '@databricks/appkit-ui/js';
import { Skeleton, useAnalyticsQuery } from '@databricks/appkit-ui/react';
import { StatusTag, runTone } from '@/components/StatusTag';

interface Props {
  ucSchema: string;
  bucket: number;
  selected: number | null;
  onSelect: (id: number) => void;
  goldenRunId: number | null;
}

/** Sidebar run list, styled like the workspace left nav: flat rows, selected row tinted. */
export function RunHistory({ ucSchema, bucket, selected, onSelect, goldenRunId }: Props) {
  const params = useMemo(() => ({ uc_schema: sql.string(ucSchema), refresh_bucket: sql.number(bucket) }), [ucSchema, bucket]);
  const { data, loading, error } = useAnalyticsQuery('runs_recent', params, { autoStart: !!ucSchema });

  return (
    <nav aria-label="Runs" className="flex min-h-0 flex-1 flex-col">
      <p className="px-3 pb-1 text-xs font-semibold text-muted-foreground">Recent runs</p>
      <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-1.5 pb-3">
        {loading && Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="mx-1.5 h-11" />)}
        {error && <p className="px-1.5 text-xs text-destructive">Could not load runs: {error}</p>}
        {!loading && !error && !data?.length && <p className="px-1.5 text-xs text-muted-foreground">No runs yet. Start one with the sample bearing.</p>}
        {data?.map((r) => {
          const id = Number(r.job_run_id);
          const { tone, label } = runTone(r.status);
          const active = selected === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => onSelect(id)}
              aria-current={active ? 'page' : undefined}
              className={`w-full rounded-md px-1.5 py-1.5 text-left transition-colors hover:bg-sidebar-accent ${
                active ? 'bg-sidebar-accent text-sidebar-accent-foreground' : ''
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate font-semibold">{r.target_class_name}</span>
                {goldenRunId === id && <span className="shrink-0 rounded-sm border px-1 text-[11px] leading-4 text-muted-foreground">Golden</span>}
              </div>
              <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                <StatusTag tone={tone} label={label} small />
                <span className="tabular-nums">
                  {r.submitted_at ? new Date(r.submitted_at).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }) : ''}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
