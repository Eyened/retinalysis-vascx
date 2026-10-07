const { test, expect } = require('@playwright/test');

test('sidebar, reliability assets and downloads work at the project base URL', async ({ page }, testInfo) => {
  await page.goto('./');
  await expect(page.getByRole('heading', { name: 'About', exact: true })).toBeVisible();
  const sidebar = page.locator('.theme-doc-sidebar-container');
  await expect(sidebar.getByText('Biomarkers & reliability', { exact: true })).toBeVisible();
  await sidebar.getByRole('link', { name: 'Macula-centered', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Macula-centered', exact: true })).toBeVisible();
  await expect(page.getByRole('cell', { name: '1.1.2.dev2+gf9ff26462', exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('macula-desktop-top.png') });
  const images = page.locator('article img');
  expect(await images.count()).toBeGreaterThanOrEqual(5);
  for (const img of await images.all()) {
    await img.scrollIntoViewIfNeeded();
    await expect.poll(() => img.evaluate(n => n.complete && n.naturalWidth > 0)).toBeTruthy();
  }
  const failures = await images.evaluateAll(nodes => nodes.filter(n => !n.complete || n.naturalWidth === 0).map(n => n.src));
  expect(failures).toEqual([]);
  const download = page.getByRole('link', { name: 'All ICC results', exact: true });
  const response = await page.request.get(await download.getAttribute('href'));
  expect(response.ok()).toBeTruthy();
  expect(await response.text()).toContain('ci_low,ci_high');
  await page.screenshot({ path: testInfo.outputPath('macula-desktop.png'), fullPage: true });
  await page.goto('biomarkers/od-centered/');
  await expect(page.getByRole('heading', { name: 'OD-centered', exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('od-desktop.png'), fullPage: true });
});

test('mobile navigation and images fit the viewport', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('biomarkers/macula-centered/');
  await page.getByRole('button', { name: 'Toggle navigation bar' }).click();
  await expect(page.locator('.navbar-sidebar')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('mobile-navigation.png') });
  await page.getByRole('button', { name: 'Close navigation bar' }).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
  await expect(page.locator('.navbar-sidebar--show')).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('macula-mobile.png'), animations: 'disabled' });
});
