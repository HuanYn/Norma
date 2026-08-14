import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  clearScreen: false,
  build: {
    outDir: "ai/web_dist",
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    port: 1420,
    strictPort: true,
    proxy: {
      "/health": "http://127.0.0.1:8765",
      "/capabilities": "http://127.0.0.1:8765",
      "/albums": "http://127.0.0.1:8765",
      "/selections": "http://127.0.0.1:8765",
      "/feedback": "http://127.0.0.1:8765",
      "/jobs": "http://127.0.0.1:8765",
      "/preferences": "http://127.0.0.1:8765",
      "/providers": "http://127.0.0.1:8765",
      "/media": "http://127.0.0.1:8765",
    },
  },
});
