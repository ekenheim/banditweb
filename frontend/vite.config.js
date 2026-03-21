import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // In dev: proxy /bandit → your cluster ingress (set VITE_DEV_CLUSTER)
      '/bandit': {
        target: process.env.VITE_DEV_CLUSTER || 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
      },
      '/analysis': {
        target: process.env.VITE_ANALYSIS_URL || 'http://localhost:8090',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/analysis/, ''),
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  }
})
