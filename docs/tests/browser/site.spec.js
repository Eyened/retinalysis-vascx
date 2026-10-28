const { test, expect } = require('@playwright/test');

test('sidebar and reliability figures work at the project base URL', async ({ page }, testInfo) => {
  await page.goto('./');
  await expect(page.getByRole('heading', { name: 'About', exact: true })).toBeVisible();
  const sidebar = page.locator('.theme-doc-sidebar-container');
  await expect(sidebar.getByText('Test-retest reliability', { exact: true })).toBeVisible();
  await sidebar.getByRole('link', { name: 'Macula-centered', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Macula-centered', exact: true })).toBeVisible();
  await expect(page.getByRole('cell', { name: 'macula_centered_rs', exact: true })).toBeVisible();
  await expect(page.locator('article')).toContainText('macula-centered (F2) images');
  await page.screenshot({ path: testInfo.outputPath('macula-desktop-top.png') });
  await expect(page.getByText('Computed combinations:', { exact: true }).first()).toBeVisible();
  await expect(page.locator('article')).not.toContainText('Retinal vessel caliber describes');
  await expect(page.locator('article img')).toHaveCount(6);
  await expect(page.getByRole('img', { name: /device-pair ICCs$/ })).toHaveCount(6);
  const images = page.locator('article img');
  expect(await images.count()).toBeGreaterThanOrEqual(6);
  for (const img of await images.all()) {
    await img.scrollIntoViewIfNeeded();
    await expect.poll(() => img.evaluate(n => n.complete && n.naturalWidth > 0)).toBeTruthy();
  }
  const failures = await images.evaluateAll(nodes => nodes.filter(n => !n.complete || n.naturalWidth === 0).map(n => n.src));
  expect(failures).toEqual([]);
  await expect(page.locator('article a[href*="/assets/"]')).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('macula-desktop.png'), fullPage: true });
  await page.goto('reliability/od-centered/');
  await expect(page.getByRole('heading', { name: 'OD-centered', exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('od-desktop.png'), fullPage: true });
});

test('mobile navigation and images fit the viewport', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('reliability/macula-centered/');
  await page.getByRole('button', { name: 'Toggle navigation bar' }).click();
  await expect(page.locator('.navbar-sidebar')).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('mobile-navigation.png') });
  await page.getByRole('button', { name: 'Close navigation bar' }).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
  await expect(page.locator('.navbar-sidebar--show')).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('macula-mobile.png'), animations: 'disabled' });
});


test('authored biomarker pages show equations and both report examples', async ({ page }, testInfo) => {
  await page.goto('biomarkers/vascular-density/');
  await expect(page.getByRole('heading', { name: 'Vascular density', exact: true })).toBeVisible();
  await expect(page.locator('.katex-display')).toHaveCount(1);
  await expect(page.locator('.katex-error')).toHaveCount(0);
  const sidebar = page.locator('.theme-doc-sidebar-container');
  await expect(sidebar.getByText('Test-retest reliability', { exact: true })).toBeVisible();
  const images = page.locator('article img');
  await expect(images).toHaveCount(2);
  for (const name of [/^Macula-centered vascular density/, /^OD-centered vascular density/]) {
    const img = page.getByRole('img', { name });
    await img.scrollIntoViewIfNeeded();
    await expect.poll(() => img.evaluate(n => n.complete && n.naturalWidth > 0)).toBeTruthy();
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath('biomarker-density-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('biomarkers/tortuosity/');
  await expect(page.locator('.katex-display')).toHaveCount(4);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
  await page.screenshot({ path: testInfo.outputPath('biomarker-tortuosity-mobile.png'), fullPage: true });
});


for (const feature of ['bifurcation', 'fractal-dimension']) {
  test(`${feature} shows both refreshed 2×3 report grids`, async ({ page }) => {
    await page.goto(`biomarkers/${feature}/`);
    const images = page.locator('article img');
    await expect(images).toHaveCount(2);
    for (const img of await images.all()) {
      await img.scrollIntoViewIfNeeded();
      await expect.poll(() => img.evaluate(n => n.complete && n.naturalWidth > n.naturalHeight)).toBeTruthy();
    }
  });
}

for (const [route, label] of [
  ['biomarkers/vascular-density/', /^Expand Macula-centered vascular density/],
  ['reliability/macula-centered/', /^Expand Bifurcation features/],
  ['reliability/overview/', /^Expand F1 paired images/],
]) {
  test(`figure modal fits the image and restores keyboard focus: ${route}`, async ({ page }) => {
    await page.goto(route);
    const trigger = page.getByRole('button', { name: label }).first();
    await trigger.scrollIntoViewIfNeeded();
    await trigger.focus();
    await page.keyboard.press('Enter');
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    const image = dialog.getByRole('img');
    await expect.poll(() => image.evaluate(n => n.complete && n.naturalWidth > 0)).toBeTruthy();
    await expect(dialog.getByRole('button', { name: 'Full resolution' })).toHaveCount(0);
    expect(await image.evaluate(n => n.getBoundingClientRect().width <= innerWidth)).toBeTruthy();
    await page.keyboard.press('Escape');
    await expect(dialog).toHaveCount(0);
    await expect(trigger).toBeFocused();
    await expect(page.locator('body')).not.toHaveCSS('overflow', 'hidden');
  });
}

test('reliability overview, concise metadata and QC guidance are linked', async ({ page }) => {
  await page.goto('./');
  const sidebar = page.locator('.theme-doc-sidebar-container');
  // Category toggles can share the destination URL; select only navigation links.
  await sidebar.getByRole('link').and(sidebar.locator('a[href$="/reliability/overview/"]')).click();
  await expect(page).toHaveURL(/\/reliability\/overview\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await expect(page.getByRole('img', { name: /paired images by device comparison$/ })).toHaveCount(2);
  await expect(page.getByRole('heading', { name: /^F1 \(OD-centered\)/ })).toBeVisible();
  await expect(page.getByRole('heading', { name: /^F2 \(Macula-centered\)/ })).toBeVisible();
  await expect(page.getByRole('columnheader', { name: 'Analysed pairs', exact: true })).toHaveCount(2);
  await page.goto('reliability/od-centered/');
  const metadata = page.locator('article table').first();
  await expect(metadata.locator('tbody tr')).toHaveCount(2);
  await expect(metadata).toContainText('VascX version');
  await expect(metadata).toContainText('Feature set');
  await expect(page.locator('article')).toContainText('optic-disc-centered (F1) images');
  await expect(page.locator('article a[href*="/assets/"]')).toHaveCount(0);
  for (const label of ['Source fingerprints and provenance', 'All ICC results', 'Pairing attrition',
                       'Study design, quality control and paired examples']) {
    await expect(page.getByRole('link', { name: label, exact: true })).toHaveCount(0);
  }
  await sidebar.getByRole('link').and(sidebar.locator('a[href$="/outputs/analysis-and-qc/"]')).click();
  await expect(page).toHaveURL(/\/outputs\/analysis-and-qc\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await expect(page.locator('article')).toContainText('EyePACS');
});

test('paired figures can be inspected and closed on mobile', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('reliability/overview/');
  await page.getByRole('button', { name: /^Expand F2 paired images/ }).click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await expect.poll(() => dialog.getByRole('img').evaluate(n => n.complete && n.naturalWidth > 0)).toBeTruthy();
  await expect(dialog.getByRole('button', { name: 'Full resolution' })).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
  await page.screenshot({ path: testInfo.outputPath('paired-modal-mobile.png') });
  await dialog.getByRole('button', { name: 'Close figure' }).click();
  await expect(dialog).toHaveCount(0);
});


test('About gallery loads all report images at the project base URL', async ({ page }, testInfo) => {
  await page.goto('./');
  const gallery = page.getByRole('region', { name: 'VascX retinal biomarker gallery' });
  await expect(gallery).toBeVisible();
  const images = gallery.locator('img');
  await expect(images).toHaveCount(6);
  for (const image of await images.all()) {
    await expect.poll(() => image.evaluate(n => n.complete && n.naturalWidth > 0)).toBeTruthy();
  }
  await page.screenshot({ path: testInfo.outputPath('about-gallery.png') });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: testInfo.outputPath('about-gallery-mobile.png') });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
});
