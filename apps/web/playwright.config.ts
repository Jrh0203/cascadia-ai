import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  outputDir: "../../artifacts/web-test-results",
  fullyParallel: false,
  retries: 0,
  reporter: [
    ["list"],
    ["html", { outputFolder: "../../artifacts/web-report", open: "never" }],
  ],
  use: {
    baseURL: "http://127.0.0.1:5187",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: [
    {
      command: "cargo run -p cascadia-api -- --api-only --listen 127.0.0.1:8787",
      cwd: "../..",
      url: "http://127.0.0.1:8787/api/v1/health",
      reuseExistingServer: true,
      timeout: 120_000,
    },
    {
      command: "npm run dev -- --port 5187",
      cwd: ".",
      url: "http://127.0.0.1:5187",
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
  projects: [
    {
      name: "desktop",
      use: {
        ...devices["Desktop Chrome"],
        channel: "chrome",
        viewport: { width: 1440, height: 960 },
      },
    },
    {
      name: "mobile",
      use: {
        ...devices["iPhone 13"],
        browserName: "chromium",
        channel: "chrome",
      },
    },
  ],
});
