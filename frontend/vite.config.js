import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 60613,
    open: true,
    proxy: {
      '/api': {
        target: 'http://localhost:43037',
        changeOrigin: true,
        secure: false
      }
    }
  }
})
