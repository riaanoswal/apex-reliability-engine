import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// run.py sets APEX_API when it has to move the backend off port 8000.
const API_TARGET = process.env.APEX_API || 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
