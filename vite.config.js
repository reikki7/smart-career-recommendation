import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  base: "/ai-job-recommendation",
  plugins: [react()],
  server: {
    allowedHosts: [
      "89b2-103-147-8-167.ngrok-free.app",
      "j6jffc7yqdr7.share.zrok.io",
    ],
  },
});
