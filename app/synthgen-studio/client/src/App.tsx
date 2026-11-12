import { useEffect, useMemo, useState } from 'react';
import { useInitialView } from '@/lib/api';
import { sql } from '@databricks/appkit-ui/js';
import {
  Alert,
  AlertDescription,
  AlertTitle,
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
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
  Skeleton,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  useAnalyticsQuery,
} from '@databricks/appkit-ui/react';
import { Database, ExternalLink, Plus } from 'lucide-react';
import { NewRunDialog } from '@/components/NewRunDialog';
import { StatusTag, type Tone } from '@/components/StatusTag';
import { RunHistory } from '@/components/RunHistory';
import { PipelineStepper } from '@/components/PipelineStepper';
import { SplatViewer } from '@/components/SplatViewer';
import { DatasetGallery } from '@/components/DatasetGallery';
import { ModelCard } from '@/components/ModelCard';
import { isTerminal, useConfig, useRefreshBucket, useRunStatus, useStageProgress, volumeUrl, type RunStatus } from '@/lib/api';

function jobTone(s: RunStatus | null): { tone: Tone; label: string } | null {
  if (!s) return null;
  if (s.queued) return { tone: 'queued', label: 'Queued' };
  if (!isTerminal(s)) return { tone: 'running', label: 'Running' };
  if (s.resultState === 'SUCCESS') return { tone: 'success', label: 'Succeeded' };
  if (s.resultState === 'CANCELED') return { tone: 'skipped', label: 'Canceled' };
  return { tone: 'failed', label: 'Failed' };
}

function readRunFromUrl() {
  const v = Number(new URLSearchParams(window.location.search).get('run'));
  return Number.isSafeInteger(v) && v > 0 ? v : null;
}

export default function App() {
  const { cfg, error: cfgError } = useConfig();
  const [pickedRunId, setRunId] = useState<number | null>(readRunFromUrl);
  // Default selection: the configured golden run, else the newest registered run (derived, not stored).
  const uc = cfg?.ucSchema ?? '';
  const recent = useAnalyticsQuery('runs_recent', useMemo(() => ({ uc_schema: sql.string(uc), refresh_bucket: sql.number(0) }), [uc]), { autoStart: !!uc });
  const defaultRunId = useMemo(() => {
    if (cfg?.goldenRunId) return cfg.goldenRunId;
    const best = recent.data?.find((r) => r.status === 'model_registered') ?? recent.data?.[0];
    return best ? Number(best.job_run_id) : null;
  }, [cfg, recent.data]);
  const runId = pickedRunId ?? defaultRunId;
  const [tab, setTab] = useState('splat');
  const [creating, setCreating] = useState(false);
  const { status, error: statusError } = useRunStatus(runId);
  const live = !!status && !isTerminal(status);
  const bucket = useRefreshBucket(live);

  const taskDone = (k: string) => status?.tasks.find((t) => t.taskKey === k)?.resultState === 'SUCCESS';
  const taskRunning = (k: string) => status?.tasks.find((t) => t.taskKey === k)?.lifeCycleState === 'RUNNING';
  const progress = {
    generate_3d_splat: useStageProgress(runId, 'splats', 'generate', live && !taskDone('generate_3d_splat')),
    render_synthetic_dataset: useStageProgress(runId, 'dataset', 'render', live && !taskDone('render_synthetic_dataset')),
    train_vision_detector: useStageProgress(runId, 'dataset', 'train', live && !taskDone('train_vision_detector')),
  };

  useEffect(() => {
    const url = new URL(window.location.href);
    if (runId) url.searchParams.set('run', String(runId));
    else url.searchParams.delete('run');
    window.history.replaceState(null, '', url);
  }, [runId]);

  const detailParams = useMemo(() => ({ uc_schema: sql.string(uc), job_run_id: sql.number(runId ?? 0), refresh_bucket: sql.number(bucket) }), [uc, runId, bucket]);
  const detail = useAnalyticsQuery('run_detail', detailParams, { autoStart: !!uc });
  const row = detail.data?.[0];

  const splatUrl = runId ? `/api/files/splats/download?path=${encodeURIComponent(`runs/${runId}/model.ply`)}` : null;
  const splatReady = taskDone('generate_3d_splat');
  const initialView = useInitialView(splatReady ? runId : null);
  const datasetReady = taskDone('render_synthetic_dataset');
  const sourceSrc = volumeUrl(cfg, 'raw', row?.source_image_path);
  const ucPath = cfg?.volumes.raw ? cfg.volumes.raw.split('/').slice(2, 4).join('.') : '';
  const isGolden = !!cfg?.goldenRunId && runId === cfg.goldenRunId;
  const tag = jobTone(status);

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b px-4">
        <img src="/databricks-logo.png" alt="" className="h-5 w-5" />
        <span className="font-semibold">SynthGen Studio</span>
        <span className="hidden text-muted-foreground sm:inline">3D synthetic data for vision models</span>
        {ucPath && (
          <span className="ml-auto inline-flex items-center gap-1.5 text-xs text-muted-foreground">
            <Database className="h-3.5 w-3.5" />
            <span className="font-mono">{ucPath}</span>
          </span>
        )}
      </header>

      <div className="flex min-h-0 flex-1 max-md:flex-col">
        <aside className="flex shrink-0 flex-col gap-3 border-sidebar-border bg-sidebar pt-3 max-md:max-h-72 max-md:border-b md:w-64 md:border-r">
          <div className="px-3">
            <Button className="w-full justify-start" onClick={() => setCreating(true)}>
              <Plus className="mr-1.5 h-4 w-4" /> New run
            </Button>
          </div>
          <RunHistory ucSchema={uc} bucket={bucket} selected={runId} onSelect={setRunId} goldenRunId={cfg?.goldenRunId ?? null} />
        </aside>

        <main className="min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[1400px] space-y-5 px-6 py-5">
          {cfgError && (
            <Alert variant="destructive">
              <AlertTitle>App configuration unavailable</AlertTitle>
              <AlertDescription>{cfgError}</AlertDescription>
            </Alert>
          )}

          {!runId ? (
            <Empty className="py-24">
              <EmptyHeader>
                <EmptyTitle>Start with one photo</EmptyTitle>
                <EmptyDescription>Use the sample bearing or upload a photo of your own part.</EmptyDescription>
              </EmptyHeader>
              <Button onClick={() => setCreating(true)}>
                <Plus className="mr-1.5 h-4 w-4" /> New run
              </Button>
            </Empty>
          ) : (
            <>
              <div className="space-y-1">
                <Breadcrumb>
                  <BreadcrumbList className="text-xs">
                    <BreadcrumbItem>
                      <BreadcrumbLink asChild>
                        <button type="button" onClick={() => setRunId(defaultRunId)}>
                          Runs
                        </button>
                      </BreadcrumbLink>
                    </BreadcrumbItem>
                    <BreadcrumbSeparator />
                    <BreadcrumbItem>
                      <BreadcrumbPage className="tabular-nums">{runId}</BreadcrumbPage>
                    </BreadcrumbItem>
                  </BreadcrumbList>
                </Breadcrumb>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex min-w-0 flex-wrap items-center gap-3">
                    {row ? (
                      <h1 className="truncate text-[22px] font-semibold leading-7">{row.target_class_name} detector</h1>
                    ) : (
                      <Skeleton className="h-7 w-56" />
                    )}
                    {tag && <StatusTag tone={tag.tone} label={tag.label} />}
                    {isGolden && <span className="rounded-sm border px-1.5 text-xs leading-5 text-muted-foreground">Golden run</span>}
                  </div>
                  {status?.runPageUrl && (
                    <Button asChild variant="outline" size="sm">
                      <a href={status.runPageUrl} target="_blank" rel="noreferrer">
                        View job run <ExternalLink className="ml-1.5 h-3.5 w-3.5" />
                      </a>
                    </Button>
                  )}
                </div>
                <p className="text-xs text-muted-foreground">
                  Run <span className="tabular-nums">{runId}</span>
                  {row?.submitted_by && <> · Submitted by {row.submitted_by}</>}
                  {status?.startTime && <> · Started {new Date(status.startTime).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}</>}
                </p>
              </div>

              {statusError && !status ? (
                <Alert variant="destructive">
                  <AlertDescription>Could not read run status: {statusError}</AlertDescription>
                </Alert>
              ) : status ? (
                <PipelineStepper status={status} progress={progress} />
              ) : (
                <Skeleton className="h-[120px] w-full" />
              )}

              <Tabs value={tab} onValueChange={setTab}>
                <TabsList>
                  <TabsTrigger value="splat">3D model</TabsTrigger>
                  <TabsTrigger value="dataset">Synthetic dataset</TabsTrigger>
                  <TabsTrigger value="model">Model</TabsTrigger>
                </TabsList>

                <TabsContent value="splat" className="mt-4">
                  <div className="grid gap-4 xl:grid-cols-[1fr_280px]">
                    {splatReady && splatUrl && initialView.ready ? (
                      <SplatViewer url={splatUrl} initialView={initialView.view} />
                    ) : (
                      <Card className="flex h-[520px] items-center justify-center">
                        <Empty>
                          <EmptyHeader>
                            <EmptyTitle>{status && isTerminal(status) ? 'No splat for this run' : 'Reconstructing in 3D…'}</EmptyTitle>
                            <EmptyDescription>
                              {progress.generate_3d_splat?.message ?? 'IDOL turns the photo into ~262k 3D Gaussians. The interactive model appears here when step 1 finishes.'}
                            </EmptyDescription>
                          </EmptyHeader>
                        </Empty>
                      </Card>
                    )}
                    <Card className="self-start">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Source photo</CardTitle>
                        <CardDescription className="text-xs">The only input. Hashed for lineage.</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        {sourceSrc ? <img src={sourceSrc} alt="Source part" className="w-full rounded-md" /> : <Skeleton className="aspect-square w-full" />}
                        <dl className="space-y-1 text-xs">
                          <div className="flex justify-between gap-2">
                            <dt className="text-muted-foreground">Gaussians</dt>
                            <dd className="tabular-nums">{row?.gaussian_count ? Number(row.gaussian_count).toLocaleString() : '—'}</dd>
                          </div>
                          <div className="flex justify-between gap-2">
                            <dt className="text-muted-foreground">IDOL inference</dt>
                            <dd className="tabular-nums">{row?.idol_seconds ? `${Number(row.idol_seconds).toFixed(1)}s` : '—'}</dd>
                          </div>
                          <div className="flex justify-between gap-2">
                            <dt className="text-muted-foreground">SHA-256</dt>
                            <dd className="truncate font-mono" title={row?.source_sha256}>
                              {row?.source_sha256?.slice(0, 12) ?? '—'}
                            </dd>
                          </div>
                        </dl>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                <TabsContent value="dataset" className="mt-4">
                  {datasetReady || (status && isTerminal(status)) ? (
                    <DatasetGallery cfg={cfg} runId={runId} bucket={bucket} className={row?.target_class_name ?? 'part'} />
                  ) : (
                    <Card className="py-16">
                      <Empty>
                        <EmptyHeader>
                          <EmptyTitle>Rendering views…</EmptyTitle>
                          <EmptyDescription>{progress.render_synthetic_dataset?.message ?? 'Starts after the splat is ready.'}</EmptyDescription>
                        </EmptyHeader>
                      </Empty>
                    </Card>
                  )}
                </TabsContent>

                <TabsContent value="model" className="mt-4">
                  <ModelCard cfg={cfg} runId={runId} bucket={bucket} liveTrain={progress.train_vision_detector} training={taskRunning('train_vision_detector')} />
                </TabsContent>
              </Tabs>
            </>
          )}
          </div>
        </main>
      </div>

      <NewRunDialog
        cfg={cfg}
        open={creating}
        onOpenChange={setCreating}
        onStarted={(id) => {
          setRunId(id);
          setTab('splat');
        }}
      />
    </div>
  );
}
