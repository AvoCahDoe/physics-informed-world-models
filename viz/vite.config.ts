import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    // three.js dominates the bundle; splitting it from the charting code lets
    // the browser fetch both in parallel and cache them independently. Only
    // /results pulls either in, so /docs and /try never pay for them.
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            { name: "three", test: /node_modules[\\/](three|@react-three)/ },
            { name: "charts", test: /node_modules[\\/](recharts|d3-|victory|react-is)/ },
          ],
        },
      },
    },
    chunkSizeWarningLimit: 900,
  },
});
