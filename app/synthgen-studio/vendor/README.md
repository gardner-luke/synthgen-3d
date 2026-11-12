Vendored browser libraries (MIT) installed as local `file:` dependencies.
The Databricks Apps build mirror does not serve `three`, so these are pinned here:
- three 0.186.1 (trimmed: core build, OrbitControls, postprocessing/Pass) — github.com/mrdoob/three.js
- @types/three 0.186.0 (transitive type deps stripped; skipLibCheck covers them)
- @sparkjsdev/spark 2.3.1 (module build + types) — github.com/sparkjsdev/spark
To refresh: npm install the versions normally, then re-run the copy step in docs/RUNBOOK.md.
