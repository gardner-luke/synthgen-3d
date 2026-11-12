import { useCallback, useRef, useState } from 'react';
import {
  Alert,
  AlertDescription,
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  Input,
  Label,
  ToggleGroup,
  ToggleGroupItem,
} from '@databricks/appkit-ui/react';
import { ImageUp, Loader2 } from 'lucide-react';
import { readError, type AppConfig } from '@/lib/api';

const SAMPLE = { rel: 'samples/bearing_fag_6304.jpg', label: 'Ball bearing (sample)', className: 'bearing' };

interface Props {
  cfg: AppConfig | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onStarted: (runId: number) => void;
}

export function NewRunDialog({ cfg, open, onOpenChange, onStarted }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [useSample, setUseSample] = useState(true);
  const [className, setClassName] = useState(SAMPLE.className);
  const [preset, setPreset] = useState<'quick' | 'full'>('full');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const pick = useCallback(
    (f: File | undefined) => {
      if (!f) return;
      if (!/^image\/(jpeg|png|webp)$/.test(f.type)) {
        setError('Use a JPG, PNG or WebP photo');
        return;
      }
      if (f.size > 25_000_000) {
        setError('Photo must be under 25 MB');
        return;
      }
      setError(null);
      setFile(f);
      setUseSample(false);
      setPreview(URL.createObjectURL(f));
      if (className === SAMPLE.className) setClassName('part');
    },
    [className],
  );

  const start = async () => {
    setError(null);
    try {
      let rel = SAMPLE.rel;
      if (!useSample && file) {
        setBusy('Uploading to Unity Catalog…');
        const ext = file.name.split('.').pop()?.toLowerCase() ?? 'jpg';
        rel = `uploads/${new Date().toISOString().slice(0, 10)}/${crypto.randomUUID()}.${ext}`;
        const up = await fetch(`/api/files/raw/upload?path=${encodeURIComponent(rel)}`, { method: 'POST', body: file });
        if (!up.ok) throw new Error(await readError(up, `Upload failed (${up.status})`));
      }
      setBusy('Starting job…');
      const r = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imagePath: rel, targetClassName: className, preset }),
      });
      const body = (await r.json().catch(() => ({}))) as { runId?: number; error?: string };
      if (!r.ok || !body.runId) throw new Error(body.error ?? `Could not start (${r.status})`);
      onStarted(body.runId);
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const p = cfg?.presets[preset];
  const sampleSrc = `/api/files/raw/raw?path=${encodeURIComponent(SAMPLE.rel)}`;

  return (
    <Dialog open={open} onOpenChange={(o) => !busy && onOpenChange(o)}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>New run</DialogTitle>
          <DialogDescription>
            One photo of a part. The job reconstructs it in 3D, renders a labeled dataset, and trains a detector registered in Unity Catalog.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-5 sm:grid-cols-[1fr_1fr]">
          <div className="space-y-1.5">
            <Label>Source photo</Label>
            <div
              role="button"
              tabIndex={0}
              aria-label="Upload a part photo"
              onClick={() => input.current?.click()}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                pick(e.dataTransfer.files?.[0]);
              }}
              className={`group relative flex aspect-[4/3] cursor-pointer items-center justify-center overflow-hidden rounded-md border border-dashed transition-colors ${
                dragging ? 'border-primary' : 'border-input hover:border-primary'
              }`}
            >
              <img src={useSample ? sampleSrc : (preview ?? sampleSrc)} alt="Selected part" className="absolute inset-0 h-full w-full object-cover" />
              <div className="relative flex items-center gap-1.5 rounded-md border bg-background px-2.5 py-1.5 opacity-0 transition-opacity group-hover:opacity-100">
                <ImageUp className="h-4 w-4 text-muted-foreground" />
                <span className="text-xs">Drop a photo or click to browse</span>
              </div>
              <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
            </div>
            <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
              <span className="truncate">{useSample ? SAMPLE.label : file?.name}</span>
              {!useSample && (
                <Button
                  type="button"
                  variant="link"
                  size="sm"
                  className="h-auto p-0 text-xs"
                  onClick={() => {
                    setUseSample(true);
                    setFile(null);
                    setClassName(SAMPLE.className);
                  }}
                >
                  Use sample
                </Button>
              )}
            </div>
          </div>

          <div className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="cls">Class name</Label>
              <Input id="cls" value={className} maxLength={40} onChange={(e) => setClassName(e.target.value)} placeholder="e.g. bearing, bracket, gear" />
              <p className="text-xs text-muted-foreground">Used as the YOLO label and in the model description.</p>
            </div>
            <div className="space-y-1.5">
              <Label>Run size</Label>
              <ToggleGroup type="single" value={preset} onValueChange={(v) => v && setPreset(v as 'quick' | 'full')} variant="outline" className="w-full">
                <ToggleGroupItem value="quick" className="flex-1">
                  Quick
                </ToggleGroupItem>
                <ToggleGroupItem value="full" className="flex-1">
                  Full
                </ToggleGroupItem>
              </ToggleGroup>
              {p && (
                <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 pt-1 text-xs">
                  <dt className="text-muted-foreground">Rendered views</dt>
                  <dd className="tabular-nums">{p.num_views}</dd>
                  <dt className="text-muted-foreground">Epochs</dt>
                  <dd className="tabular-nums">{p.epochs}</dd>
                  <dt className="text-muted-foreground">Gaussians</dt>
                  <dd className="tabular-nums">{(p.gaussian_count / 1024).toFixed(0)}k</dd>
                  <dt className="text-muted-foreground">Image size</dt>
                  <dd className="tabular-nums">{p.image_size}px</dd>
                </dl>
              )}
            </div>
          </div>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={!!busy}>
            Cancel
          </Button>
          <Button onClick={() => void start()} disabled={!!busy || !className.trim() || (!useSample && !file)}>
            {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {busy ?? 'Start run'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
