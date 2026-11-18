import { useState } from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@databricks/appkit-ui/react';
import { Ban, Loader2, MoreHorizontal, Trash2 } from 'lucide-react';
import { cancelRun, deleteRun } from '@/lib/api';

interface Props {
  runId: number;
  label: string;
  /** Run is queued or in progress (can be canceled, not deleted). */
  active: boolean;
  /** Sample runs are protected. */
  protectedRun?: boolean;
  onChanged: (kind: 'canceled' | 'deleted', runId: number) => void;
  /** Smaller trigger for sidebar rows. */
  compact?: boolean;
}

/** "..." menu with Cancel / Delete, each behind a confirmation. */
export function RunActions({ runId, label, active, protectedRun = false, onChanged, compact = false }: Props) {
  const [confirm, setConfirm] = useState<'cancel' | 'delete' | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (protectedRun && !active) return null;

  const run = async () => {
    if (!confirm) return;
    setBusy(true);
    setError(null);
    try {
      if (confirm === 'cancel') await cancelRun(runId);
      else await deleteRun(runId);
      onChanged(confirm === 'cancel' ? 'canceled' : 'deleted', runId);
      setConfirm(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant={compact ? 'ghost' : 'outline'}
            size="sm"
            aria-label={`Actions for ${label}`}
            className={compact ? 'h-6 w-6 p-0 text-muted-foreground' : ''}
            onClick={(e) => e.stopPropagation()}
          >
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          {active ? (
            <DropdownMenuItem onSelect={() => setConfirm('cancel')}>
              <Ban className="mr-2 h-4 w-4" /> Cancel run
            </DropdownMenuItem>
          ) : (
            <DropdownMenuItem onSelect={() => setConfirm('delete')} className="text-destructive focus:text-destructive">
              <Trash2 className="mr-2 h-4 w-4" /> Delete run
            </DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <AlertDialog open={!!confirm} onOpenChange={(o) => !busy && !o && (setConfirm(null), setError(null))}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{confirm === 'cancel' ? `Cancel the ${label} run?` : `Delete the ${label} run?`}</AlertDialogTitle>
            <AlertDialogDescription>
              {confirm === 'cancel'
                ? 'The job stops at its current step. Anything already written stays until you delete the run.'
                : 'This removes the run from the app: its rows in pipeline_runs, synthetic_views and training_epochs, the 3D model and the rendered dataset. A model version it registered in Unity Catalog is kept.'}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {error && <p className="text-xs text-destructive">{error}</p>}
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>Keep</AlertDialogCancel>
            <AlertDialogAction
              disabled={busy}
              onClick={(e) => {
                e.preventDefault();
                void run();
              }}
              className={confirm === 'delete' ? 'bg-destructive text-destructive-foreground hover:bg-destructive/90' : ''}
            >
              {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {confirm === 'cancel' ? 'Cancel run' : 'Delete'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
