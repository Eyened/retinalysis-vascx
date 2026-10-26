import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('the sidebar explicitly owns both top-level biomarker sections', async () => {
  const { default: sidebar } = await import('../sidebars.js');
  const biomarkers = sidebar.docs.find(item => item.label === 'Biomarkers');
  const reliability = sidebar.docs.find(item => item.label === 'Test-retest reliability');
  assert.ok(biomarkers && reliability);
  assert.ok(sidebar.docs.indexOf(reliability) < sidebar.docs.indexOf(biomarkers));
  assert.deepEqual(biomarkers.link, { type: 'doc', id: 'biomarkers/overview' });
  const ids = [biomarkers.link.id, ...biomarkers.items.flatMap(item =>
    item.type === 'doc' ? [item.id] : item.items)];
  assert.ok(biomarkers.items.filter(item => item.type === 'category').every(item => item.items.length > 1));
  assert.ok(ids.every(id => id.startsWith('biomarkers/')));
  assert.deepEqual(biomarkers.items.filter(item => typeof item !== 'string').map(item => item.label),
    ['Bifurcation features', 'CREs, diameters, and AVR', 'Tortuosity', 'Optic disc morphology', 'Image quality', 'Other']);
  assert.ok(!ids.includes('biomarkers/disc-fovea-distance'));
  assert.deepEqual(reliability.link, { type: 'doc', id: 'reliability/overview' });
  assert.ok(fs.existsSync(path.join(root, 'docs/outputs/analysis-and-qc.md')));
  assert.deepEqual(reliability.items, ['reliability/macula-centered', 'reliability/od-centered']);
  const generated = JSON.parse(fs.readFileSync(path.join(root, 'generated-assets.json')));
  for (const id of ids) {
    assert.ok(fs.existsSync(path.join(root, `docs/${id}.md`)));
    assert.ok(!Object.hasOwn(generated, `docs/${id}.md`));
  }
});


test('page and component asset URLs resolve locally', () => {
  for (const folder of ['docs', 'src']) {
    const base = path.join(root, folder);
    const files = fs.readdirSync(base, {recursive: true}).filter(name => /\.(mdx?|jsx?|tsx?|css)$/.test(name));
    for (const name of files) {
      const text = fs.readFileSync(path.join(base, name), 'utf8');
      for (const [asset] of text.matchAll(/\/assets\/[A-Za-z0-9_./%-]+/g)) {
        assert.ok(fs.existsSync(path.join(root, 'static', asset)), `${folder}/${name}: missing ${asset}`);
      }
    }
  }
});


test('the reliability overview is authored and imports generated pair counts', () => {
  const page = fs.readFileSync(path.join(root, 'docs/reliability/overview.md'), 'utf8');
  const generated = JSON.parse(fs.readFileSync(path.join(root, 'generated-assets.json')));
  assert.ok(!Object.hasOwn(generated, 'docs/reliability/overview.md'));
  assert.match(page, /import DevicePairCounts from '@site\/static\/assets\/reliability\/device-pair-counts.mdx'/);
  assert.match(page, /<DevicePairCounts \/>/);
  assert.ok(Object.hasOwn(generated, 'static/assets/reliability/device-pair-counts.mdx'));
});
