import { useMemo } from 'react';
import { sql } from '@databricks/appkit-ui/js';
import {
  Alert,
  AlertDescription,
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
  LineChart,
  Skeleton,
  useAnalyticsQuery,
} from '@databricks/appkit-ui/react';
import { ExternalLink, ShieldCheck } from 'lucide-react';
import { StatusTag } from '@/components/StatusTag';
import { fmtDuration, fmtPct, volumeUrl, type AppConfig, type EpochPoint, type StageProgress } from '@/lib/api';

interface Props {
  cfg: AppConfig | null;
  runId: number;
  bucket: number;
  liveTrain: StageProgress | null;
  training: boolean;
}

function Kpi({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-md border p-3">
      <p className="text-xs font-semibold text-muted-foreground">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

export function ModelCard({ cfg, runId, bucket, liveTrain, training }: Props) {
  const uc = cfg?.ucSchema ?? '';
  const params = useMemo(() => ({ uc_schema: sql.string(uc), job_run_id: sql.number(runId), refresh_bucket: sql.number(bucket) }), [uc, runId, bucket]);
  const detail = useAnalyticsQuery('run_detail', params, { autoStart: !!uc });
  const epochs = useAnalyticsQuery('run_epochs', params, { autoStart: !!uc });
  const row = detail.data?.[0];

  const curve: EpochPoint[] = useMemo(() => {
    const fromDelta = (epochs.data ?? []).map((e) => ({
      epoch: Number(e.epoch),
      map50: Number(e.map50),
      map50_95: Number(e.map50_95),
      precision: Number(e.precision),
      recall: Number(e.recall),
    }));
    return fromDelta.length ? fromDelta : (liveTrain?.history ?? []);
  }, [epochs.data, liveTrain]);

  // AppKit title-cases series keys ("mAP@50" -> "M Ap@50"), so the chart's legend is hidden and drawn below with the real names.
  const chartData = useMemo(() => curve.map((c) => ({ epoch: c.epoch, map50: c.map50, map50_95: c.map50_95 })), [curve]);
  const last = curve[curve.length - 1];
  const registered = row?.status === 'model_registered';
  const metric = (v: number | undefined, live: number | undefined) => fmtPct(registered ? v : live);

  if (detail.error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>Could not load the model card: {detail.error}</AlertDescription>
      </Alert>
    );
  }
  if (detail.loading) return <Skeleton className="h-80 w-full" />;
  if (!registered && !training && !curve.length) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>No model yet</EmptyTitle>
          <EmptyDescription>Metrics stream in here while step 3 trains, then the registered Unity Catalog version appears.</EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  const modelUrl =
    cfg?.host && row?.model_name && row?.model_version
      ? `${cfg.host}/explore/data/models/${row.model_name.split('.').join('/')}/version/${row.model_version}`
      : null;
  const realSrc = volumeUrl(cfg, 'dataset', row?.real_test_image_path);
  const evalHint = registered ? 'Validation split, final epoch' : `Live, epoch ${last?.epoch ?? 0}`;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex flex-row items-start justify-between gap-4 space-y-0">
          <div>
            <CardTitle className="text-[15px]">
              {registered ? `Detector registered as ${row?.model_name?.split('.').pop()} v${row?.model_version}` : 'Training detector…'}
            </CardTitle>
            <CardDescription>
              YOLO trained only on synthetic renders. Tracked in MLflow, governed in Unity Catalog with lineage back to the source photo.
            </CardDescription>
          </div>
          {registered ? (
            <div className="flex shrink-0 flex-col items-end gap-2">
              <Badge variant="outline" className="gap-1 font-mono font-normal">
                <ShieldCheck className="h-3.5 w-3.5 text-success" /> @champion · v{row?.model_version}
              </Badge>
              {modelUrl && (
                <Button asChild size="sm" variant="outline">
                  <a href={modelUrl} target="_blank" rel="noreferrer">
                    Open in Unity Catalog <ExternalLink className="ml-1 h-3.5 w-3.5" />
                  </a>
                </Button>
              )}
            </div>
          ) : (
            <StatusTag tone="running" label="Training" />
          )}
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Kpi label="mAP@50" value={metric(row?.map50, last?.map50)} hint={evalHint} />
            <Kpi label="mAP@50-95" value={metric(row?.map50_95, last?.map50_95)} hint={evalHint} />
            <Kpi label="Precision" value={metric(row?.precision, last?.precision)} hint={evalHint} />
            <Kpi label="Recall" value={metric(row?.recall, last?.recall)} hint={evalHint} />
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            Source: MLflow run {row?.mlflow_run_id ? <code>{row.mlflow_run_id.slice(0, 8)}</code> : '(pending)'} mirrored to{' '}
            <code>pipeline_runs</code> / <code>training_epochs</code>. Note: validation views are synthetic too. The real-photo check below is the honest sim→real signal.
          </p>
        </CardContent>
      </Card>

      <div className="grid items-start gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="pb-0">
            <CardTitle className="text-sm">Validation mAP by epoch</CardTitle>
          </CardHeader>
          <CardContent>
            {chartData.length ? (
              <>
                <LineChart data={chartData} xKey="epoch" yKey={['map50', 'map50_95']} height={240} colorPalette="categorical" showLegend={false} showSymbol={false} />
                <div className="flex justify-center gap-4 text-xs text-muted-foreground">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-0.5 w-4" style={{ background: 'var(--chart-cat-1)' }} /> mAP@50
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-0.5 w-4" style={{ background: 'var(--chart-cat-2)' }} /> mAP@50-95
                  </span>
                </div>
              </>
            ) : (
              <Skeleton className="h-[240px] w-full" />
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Sim → real check</CardTitle>
            <CardDescription className="text-xs">The new detector run on the original photo it has never seen.</CardDescription>
          </CardHeader>
          <CardContent>
            {realSrc ? (
              <>
                <img src={realSrc} alt="Detector prediction on the original photo" className="w-full rounded-md" />
                <p className="mt-2 text-xs text-muted-foreground">
                  Top confidence: <span className="font-medium text-foreground">{fmtPct(row?.real_test_max_conf, 0)}</span>
                </p>
              </>
            ) : (
              <p className="text-xs text-muted-foreground">Appears when training finishes.</p>
            )}
          </CardContent>
        </Card>
      </div>

      {registered && (
        <p className="text-xs text-muted-foreground">
          Timing: IDOL {fmtDuration((row?.idol_seconds ?? 0) * 1000)} · render {fmtDuration((row?.render_seconds ?? 0) * 1000)} · train{' '}
          {fmtDuration((row?.train_seconds ?? 0) * 1000)} · {row?.num_views} views · {Number(row?.gaussian_count ?? 0).toLocaleString()} Gaussians
        </p>
      )}
    </div>
  );
}
