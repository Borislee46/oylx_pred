import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsDir: "assets",
  },
  server: { port: 5185, strictPort: true },
  preview: { port: 5186, strictPort: true },
});
