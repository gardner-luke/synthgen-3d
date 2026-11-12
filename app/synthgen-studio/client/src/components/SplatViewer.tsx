import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh, SplatFileType } from '@sparkjsdev/spark';
import { Button, Progress, Alert, AlertDescription, AlertTitle } from '@databricks/appkit-ui/react';
import { FlipVertical2, Pause, Play, RotateCcw } from 'lucide-react';

interface Props {
  /** Same-origin Files plugin URL for the .ply (streams from the splat_models UC volume). */
  url: string;
  height?: number;
  /** IDOL's predicted input camera (radians). When present the viewer opens on the photo's viewpoint. */
  initialView?: { yaw: number; pitch: number } | null;
}

/**
 * Interactive 3D Gaussian splat viewer.
 * three.js + Spark (World Labs, MIT). Spark sorts splats on the GPU and does not need
 * SharedArrayBuffer, so no COOP/COEP headers are required behind the Databricks Apps proxy.
 */
export function SplatViewer({ url, height = 520, initialView = null }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{ reset: () => void; setAuto: (on: boolean) => void; flip: () => void } | null>(null);
  const [loaded, setLoaded] = useState(0);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [auto, setAuto] = useState(true);
  const [stats, setStats] = useState<{ mb: number; ms: number } | null>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let disposed = false;
    setReady(false);
    setError(null);
    setLoaded(0);

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, height);
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, el.clientWidth / height, 0.01, 1000);
    camera.position.set(0, 0.4, 2.2);
    scene.add(new SparkRenderer({ renderer }));

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.6;

    let mesh: SplatMesh | null = null;
    const home = { pos: camera.position.clone(), target: new THREE.Vector3() };

    const resize = () => {
      const w = el.clientWidth;
      renderer.setSize(w, height);
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);

    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    api.current = {
      reset: () => {
        camera.position.copy(home.pos);
        controls.target.copy(home.target);
        controls.update();
      },
      setAuto: (on) => {
        controls.autoRotate = on;
      },
      flip: () => {
        if (mesh) mesh.rotation.x += Math.PI;
      },
    };

    void (async () => {
      const t0 = performance.now();
      try {
        const res = await fetch(url);
        if (!res.ok || !res.body) throw new Error(`Could not load splat (${res.status})`);
        const total = Number(res.headers.get('content-length')) || 0;
        const reader = res.body.getReader();
        const chunks: Uint8Array[] = [];
        let got = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          got += value.length;
          if (total && !disposed) setLoaded(Math.round((got / total) * 100));
        }
        const bytes = new Uint8Array(got);
        let off = 0;
        for (const c of chunks) {
          bytes.set(c, off);
          off += c.length;
        }
        if (disposed) return;
        mesh = new SplatMesh({ fileBytes: bytes, fileType: SplatFileType.PLY });
        await mesh.initialized;
        if (disposed) return;
        scene.add(mesh);
        // Frame the object from its splat-center bounding box.
        const box = mesh.getBoundingBox(true);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();
        const dist = (size / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.7;
        controls.target.copy(center);
        const dir = initialView
          ? new THREE.Vector3(
              Math.cos(initialView.pitch) * Math.sin(initialView.yaw),
              -Math.sin(initialView.pitch),
              Math.cos(initialView.pitch) * Math.cos(initialView.yaw),
            )
          : new THREE.Vector3(0.6, 0.25, 0.75);
        camera.position.copy(center).add(dir.normalize().multiplyScalar(dist));
        camera.near = size / 100;
        camera.far = size * 100;
        camera.updateProjectionMatrix();
        home.pos.copy(camera.position);
        home.target.copy(center);
        setStats({ mb: got / 1e6, ms: performance.now() - t0 });
        setReady(true);
      } catch (e) {
        if (!disposed) setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      disposed = true;
      ro.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      mesh?.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      api.current = null;
    };
  }, [url, height, initialView]);

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Splat failed to load</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-lg border bg-muted">
      <div ref={host} style={{ height }} className="w-full cursor-grab active:cursor-grabbing" />
      {!ready && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-background/70">
          <p className="text-sm text-muted-foreground">Streaming splat from Unity Catalog…</p>
          <Progress value={loaded} className="w-56" />
        </div>
      )}
      {ready && (
        <>
          <div className="absolute top-3 right-3 flex gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                api.current?.setAuto(!auto);
                setAuto(!auto);
              }}
            >
              {auto ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              <span className="ml-1">{auto ? 'Pause' : 'Spin'}</span>
            </Button>
            <Button size="sm" variant="outline" onClick={() => api.current?.reset()}>
              <RotateCcw className="h-4 w-4" />
              <span className="ml-1">Reset</span>
            </Button>
            <Button size="sm" variant="outline" onClick={() => api.current?.flip()} title="Flip if the model loads upside down">
              <FlipVertical2 className="h-4 w-4" />
            </Button>
          </div>
          <p className="absolute bottom-3 left-3 rounded-md border bg-background px-2 py-0.5 text-xs text-muted-foreground">
            Drag to orbit · scroll to zoom · right-drag to pan
            {stats && ` · ${stats.mb.toFixed(1)} MB in ${(stats.ms / 1000).toFixed(1)}s`}
          </p>
        </>
      )}
    </div>
  );
}
