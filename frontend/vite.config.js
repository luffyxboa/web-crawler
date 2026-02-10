import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  preview: {
    port: process.env.PORT || 5173,
    host: '0.0.0.0',
    strictPort: false,
    allowedHosts: ['leadgen.up.railway.app', 'localhost'],
  }
})
