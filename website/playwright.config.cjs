const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: './tests/browser',
  use: { baseURL: 'http://127.0.0.1:3105/retinalysis-vascx/', headless: true },
  webServer: { command: 'npm run serve -- --port 3105 --host 127.0.0.1',
    url: 'http://127.0.0.1:3105/retinalysis-vascx/', reuseExistingServer: !process.env.CI },
  reporter: 'list',
});
