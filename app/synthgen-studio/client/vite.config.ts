import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';

// https://vite.dev/config/
export default defineConfig({
  root: __dirname,
  plugins: [react(), tailwindcss()],
  server: {
    middlewareMode: true,
  },
  build: {
    outDir: path.resolve(__dirname, './dist'),
    emptyOutDir: true,
    sourcemap: process.env.NODE_ENV === 'development',
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'react/jsx-dev-runtime', 'react/jsx-runtime'],
  },
  resolve: {
    // three.js + Spark are vendored (MIT) because the Databricks Apps npm mirror doesn't serve `three`.
    // Order matters: subpath aliases before the bare `three` alias.
    alias: [
      { find: '@', replacement: path.resolve(__dirname, './src') },
      { find: /^three\/(examples\/jsm|addons)\/(.*)$/, replacement: path.resolve(__dirname, '../vendor/three/examples/jsm') + '/$2' },
      { find: /^three$/, replacement: path.resolve(__dirname, '../vendor/three/lib/three.module.js') },
      { find: /^@sparkjsdev\/spark$/, replacement: path.resolve(__dirname, '../vendor/spark/lib/spark.module.js') },
    ],
  },
});
