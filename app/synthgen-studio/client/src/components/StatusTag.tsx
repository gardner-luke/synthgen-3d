import { CheckCircle2, CircleDashed, Clock, Loader2, MinusCircle, XCircle } from 'lucide-react';

export type Tone = 'success' | 'running' | 'failed' | 'queued' | 'pending' | 'skipped';

const ICON = {
  success: { Icon: CheckCircle2, cls: 'text-success' },
  running: { Icon: Loader2, cls: 'text-primary animate-spin' },
  failed: { Icon: XCircle, cls: 'text-destructive' },
  queued: { Icon: Clock, cls: 'text-muted-foreground' },
  pending: { Icon: CircleDashed, cls: 'text-muted-foreground' },
  skipped: { Icon: MinusCircle, cls: 'text-muted-foreground' },
} as const;

/** Icon + label status, the way the Jobs run list shows run state (no filled pill). */
export function StatusTag({ tone, label, small = false, className = '' }: { tone: Tone; label: string; small?: boolean; className?: string }) {
  const { Icon, cls } = ICON[tone];
  return (
    <span className={`inline-flex items-center whitespace-nowrap ${small ? 'gap-1 text-xs' : 'gap-1.5 text-[13px]'} ${className}`}>
      <Icon className={`shrink-0 ${small ? 'h-3.5 w-3.5' : 'h-4 w-4'} ${cls}`} aria-hidden />
      {label}
    </span>
  );
}

/** pipeline_runs.status -> tag. */
export function runTone(status: string | null | undefined): { tone: Tone; label: string } {
  switch (status) {
    case 'model_registered':
      return { tone: 'success', label: 'Succeeded' };
    case 'failed':
      return { tone: 'failed', label: 'Failed' };
    case 'canceled':
      return { tone: 'skipped', label: 'Canceled' };
    case 'running_generate':
      return { tone: 'running', label: 'Generating splat' };
    case 'splat_ready':
    case 'running_render':
      return { tone: 'running', label: 'Rendering' };
    case 'dataset_ready':
    case 'running_train':
      return { tone: 'running', label: 'Training' };
    default:
      return { tone: 'pending', label: status ?? 'Pending' };
  }
}
