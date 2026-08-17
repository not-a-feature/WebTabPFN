import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./browser",
  timeout: 120_000,
  fullyParallel: false,
  use: {
    baseURL: "http://127.0.0.1:4173",
    browserName: "chromium",
    headless: true,
  },
  webServer: {
    command: "python -m http.server 4173 --directory ..",
    url: "http://127.0.0.1:4173/benchmark/",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
