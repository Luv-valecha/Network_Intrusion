import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(),
  tailwindcss(),
  ],
  server: {
    proxy: {
      '/predict': {
        target: 'http://35.223.135.16:5000',
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
