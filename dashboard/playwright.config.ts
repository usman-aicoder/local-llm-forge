import { defineConfig, devices } from "@playwright/test"

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  retries: 0,

  use: {
    baseURL: "http://localhost:3010",
    headless: true,
    screenshot: "only-on-failure",
    video: "off",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  // Start dev server automatically when running tests
  webServer: {
    command: "npm run dev -- --port 3010",
    url: "http://localhost:3010",
    reuseExistingServer: true,    // Use already-running dev server if present
    timeout: 30_000,
  },
})
