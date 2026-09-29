import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { root, renderPage } from './readme.mjs';
const source = fs.readFileSync(path.join(root, '../README.md'), 'utf8');
for (const page of JSON.parse(fs.readFileSync(path.join(root, 'content-map.json')))) {
  const actual = fs.readFileSync(path.join(root, `docs/${page.id}.md`), 'utf8');
  if (actual !== renderPage(source, page)) throw new Error(`README content drift: ${page.id}`);
}
const assets = JSON.parse(fs.readFileSync(path.join(root, 'generated-assets.json')));
for (const [relative, expected] of Object.entries(assets)) {
  if (!/^(docs\/biomarkers\/|static\/assets\/reliability\/)/.test(relative) || relative.includes('..'))
    throw new Error(`Unexpected generated path: ${relative}`);
  const content = fs.readFileSync(path.join(root, relative));
  if (createHash('sha256').update(content).digest('hex') !== expected)
    throw new Error(`Generated asset drift: ${relative}`);
  if (/\.(json|csv|md)$/.test(relative) && /\/home\/|\/mnt\/|\/shared\/|report_sample_ids|"image_id"|"patient_id"/.test(content.toString()))
    throw new Error(`Private data in generated asset: ${relative}`);
}
console.log('README content and generated assets validated.');
