import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const handleProxyError = (proxy) => {
  proxy.on('error', (err, _req, res) => {
    // If backend is still initializing or disconnected
    if (err.code === 'ECONNREFUSED' || err.code === 'ECONNABORTED' || err.code === 'ECONNRESET') {
      if (res && !res.headersSent && typeof res.writeHead === 'function') {
        res.writeHead(503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ detail: 'Backend server is initializing or unavailable' }));
      }
      return;
    }
    console.warn('[vite proxy error]', err.message);
  });
};

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    allowedHosts: true,
    proxy: {
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        configure: handleProxyError,
      },
      '/employees': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        configure: handleProxyError,
      },
      '/monitoring': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        configure: handleProxyError,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
        configure: (proxy) => {
          proxy.on('error', (err) => {
            // Ignore socket disconnects/aborts on browser reload or backend warmup
            if (err.code !== 'ECONNABORTED' && err.code !== 'ECONNRESET' && err.code !== 'ECONNREFUSED') {
              console.warn('[vite ws proxy error]', err.message);
            }
          });
        },
      },
    },
  },
})

