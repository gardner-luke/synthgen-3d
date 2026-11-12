import { useMemo, useState } from 'react';
import { sql } from '@databricks/appkit-ui/js';
import {
  Alert,
  AlertDescription,
  Badge,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Dialog,
  DialogContent,
  DialogTitle,
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
  Skeleton,
  useAnalyticsQuery,
} from '@databricks/appkit-ui/react';
import { volumeUrl, type AppConfig } from '@/lib/api';

/** Analytics JSON can deliver ARRAY columns as a JSON string; accept either shape. */
function parseNumberArray(v: unknown): number[] {
  let arr: unknown = v;
  if (typeof v === 'string') {
    try {
      arr = JSON.parse(v);
    } catch {
      return [];
    }
  }
  return Array.isArray(arr) ? arr.map(Number).filter((n) => Number.isFinite(n)) : [];
}

interface View {
  view_id: number;
  split: string;
  image_path: string;
  bbox: [number, number, number, number] | null;
  width: number;
  height: number;
  az: number;
  el: number;
  bg: string;
}

function BoxedImage({ v, src, label, big = false, eager = false }: { v: View; src: string; label: string; big?: boolean; eager?: boolean }) {
  const [x0, y0, x1, y1] = v.bbox ?? [0, 0, 0, 0];
  return (
    <div className="relative overflow-hidden rounded-md bg-muted" style={{ aspectRatio: `${v.width} / ${v.height}` }}>
      <img src={src} alt={`Synthetic view ${v.view_id}`} loading={eager ? "eager" : "lazy"} decoding="async" className="absolute inset-0 h-full w-full object-cover" />
      {v.bbox && (
        <svg viewBox={`0 0 ${v.width} ${v.height}`} className="pointer-events-none absolute inset-0 h-full w-full">
          {(() => {
            const fs = big ? 18 : 34;
            const pad = big ? 6 : 10;
            const tagH = fs + pad;
            const tagY = y0 - tagH >= 0 ? y0 - tagH : y0;
            return (
              <>
                <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke="var(--chart-1)" strokeWidth={big ? 3 : 6} />
                <rect x={x0} y={tagY} width={label.length * fs * 0.62 + pad * 2} height={tagH} fill="var(--chart-1)" />
                <text x={x0 + pad} y={tagY + fs * 0.92} fill="var(--primary-foreground)" fontSize={fs} fontWeight={600} fontFamily="var(--font-db)">
                  {label}
                </text>
              </>
            );
          })()}
        </svg>
      )}
    </div>
  );
}

export function DatasetGallery({ cfg, runId, bucket, className }: { cfg: AppConfig | null; runId: number; bucket: number; className: string }) {
  const uc = cfg?.ucSchema ?? '';
  const params = useMemo(() => ({ uc_schema: sql.string(uc), job_run_id: sql.number(runId), refresh_bucket: sql.number(bucket) }), [uc, runId, bucket]);
  const { data, loading, error } = useAnalyticsQuery('run_views', params, { autoStart: !!uc });
  const stats = useAnalyticsQuery('run_view_stats', params, { autoStart: !!uc });
  const [open, setOpen] = useState<View | null>(null);

  const views: View[] = useMemo(
    () =>
      (data ?? []).map((r) => {
        const b = parseNumberArray(r.bbox_xyxy);
        return {
          view_id: Number(r.view_id),
          split: r.split,
          image_path: r.image_path,
          bbox: b.length === 4 ? (b as [number, number, number, number]) : null,
          width: Number(r.width),
          height: Number(r.height),
          az: Number(r.azimuth_deg),
          el: Number(r.elevation_deg),
          bg: r.background_kind,
        };
      }),
    [data],
  );

  const mix = useMemo(() => {
    const by: Record<string, number> = {};
    for (const r of stats.data ?? []) by[r.background_kind] = (by[r.background_kind] ?? 0) + Number(r.n);
    return Object.entries(by).map(([background, views]) => ({ background, views }));
  }, [stats.data]);
  const split = useMemo(() => {
    const by: Record<string, number> = {};
    for (const r of stats.data ?? []) by[r.split] = (by[r.split] ?? 0) + Number(r.n);
    return by;
  }, [stats.data]);

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>Could not load the dataset: {error}</AlertDescription>
      </Alert>
    );
  }
  if (loading) {
    return (
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-6">
        {Array.from({ length: 12 }).map((_, i) => (
          <Skeleton key={i} className="aspect-square w-full" />
        ))}
      </div>
    );
  }
  if (!views.length) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>No synthetic views yet</EmptyTitle>
          <EmptyDescription>Renders show up here once step 2 finishes writing the dataset to Unity Catalog.</EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid items-start gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-[15px]">
              {(split.train ?? 0) + (split.val ?? 0)} labeled images, zero manual annotation
            </CardTitle>
            <CardDescription>
              {split.train ?? 0} train · {split.val ?? 0} val · boxes come from the alpha mask of each render, so every label is pixel-exact.
              Source: <code className="text-xs">synthetic_views</code> (Delta, Unity Catalog).
            </CardDescription>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="pb-0">
            <CardTitle className="text-sm">Background randomization</CardTitle>
          </CardHeader>
          <CardContent>
            {stats.loading ? (
              <Skeleton className="h-[120px] w-full" />
            ) : (
              <ul className="mt-3 space-y-2">
                {mix.map((m) => {
                  const max = Math.max(...mix.map((x) => x.views), 1);
                  return (
                    <li key={m.background} className="grid grid-cols-[72px_1fr_32px] items-center gap-2 text-xs">
                      <span className="text-muted-foreground">{m.background}</span>
                      <span className="h-1.5 rounded-sm bg-muted">
                        <span className="block h-1.5 rounded-sm bg-primary" style={{ width: `${(m.views / max) * 100}%` }} />
                      </span>
                      <span className="text-right tabular-nums">{m.views}</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-6">
        {views.map((v) => {
          const src = volumeUrl(cfg, 'dataset', v.image_path);
          if (!src) return null;
          return (
            <button key={v.view_id} type="button" onClick={() => setOpen(v)} className="group text-left">
              <BoxedImage v={v} src={src} label={className} eager={v.view_id < 12} />
              <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
                <span>#{v.view_id}</span>
                <span className="rounded-sm border px-1 leading-4">{v.split}</span>
              </div>
            </button>
          );
        })}
      </div>
      <p className="text-xs text-muted-foreground">Showing the first {views.length} views. The full set lives in the synthetic_dataset volume, YOLO format.</p>

      <Dialog open={!!open} onOpenChange={(o) => !o && setOpen(null)}>
        <DialogContent className="max-w-3xl">
          <DialogTitle>View #{open?.view_id}</DialogTitle>
          {open && (
            <>
              <BoxedImage v={open} src={volumeUrl(cfg, 'dataset', open.image_path) ?? ''} label={className} big />
              <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                <Badge variant="outline">azimuth {open.az.toFixed(0)}°</Badge>
                <Badge variant="outline">elevation {open.el.toFixed(0)}°</Badge>
                <Badge variant="outline">background: {open.bg}</Badge>
                <Badge variant="outline">split: {open.split}</Badge>
                {open.bbox && <Badge variant="outline">bbox [{open.bbox.join(', ')}]</Badge>}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
