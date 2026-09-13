import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { sseDemoPlugin } from "./sseDemoPlugin";

export default defineConfig({
  plugins: [react(), sseDemoPlugin()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    assetsDir: "assets",
  },
});
