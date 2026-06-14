import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5187,
    strictPort: false,
    proxy: {
      "/api": "http://127.0.0.1:8787",
    },
  },
  build: {
    sourcemap: true,
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
