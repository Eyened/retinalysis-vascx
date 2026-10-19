import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const assets = JSON.parse(fs.readFileSync(path.join(root, 'generated-assets.json')));
for (const [relative, expected] of Object.entries(assets)) {
  if (!/^(docs\/reliability\/|static\/assets\/(?:reliability|biomarkers)\/)/.test(relative) || relative.includes('..'))
    throw new Error(`Unexpected generated path: ${relative}`);
  const content = fs.readFileSync(path.join(root, relative));
  if (createHash('sha256').update(content).digest('hex') !== expected)
    throw new Error(`Generated asset drift: ${relative}`);
  if (/\.(json|csv|mdx?)$/.test(relative) && /\/home\/|\/mnt\/|\/shared\/|report_sample_ids|"image_id"|"patient_id"/.test(content.toString()))
    throw new Error(`Private data in generated asset: ${relative}`);
}
console.log('Generated assets validated.');
