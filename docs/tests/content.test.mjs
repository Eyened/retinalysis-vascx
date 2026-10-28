import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('authored biomarker pages are not owned by the asset generator', () => {
  const generated = JSON.parse(fs.readFileSync(path.join(root, 'generated-assets.json')));
  const files = fs.readdirSync(path.join(root, 'docs/biomarkers'), {recursive: true})
    .filter(name => /\.mdx?$/.test(name));
  for (const name of files) {
    const relative = `docs/biomarkers/${name.split(path.sep).join('/')}`;
    assert.ok(!Object.hasOwn(generated, relative), `${relative} must remain manually maintained`);
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
