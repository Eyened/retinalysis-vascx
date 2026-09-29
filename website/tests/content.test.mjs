import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { root, sections, selectBlock, renderPage, relocateLinks } from '../scripts/readme.mjs';

test('headings inside fenced code do not split source sections', () => {
  const result = sections('## A\n\n```\n## B\n```\n\n## C\ntext');
  assert.deepEqual([...result.keys()], ['A', 'C']);
});
test('selectors fail on missing or ambiguous boundaries', () => {
  assert.throws(() => selectBlock(new Map(), {section:'absent'}));
  assert.throws(() => selectBlock(new Map([['A','xx xx']]), {section:'A',from:'xx'}));
});
test('link relocation preserves code examples', () => {
  const source = '[Example](./notebooks/a.ipynb)\n\n`[Example](./notebooks/a.ipynb)`';
  assert.equal(relocateLinks(source), '[Example](https://github.com/Eyened/retinalysis-vascx/blob/main/notebooks/a.ipynb)\n\n`[Example](./notebooks/a.ipynb)`');
});
test('every README-derived page matches configured source blocks', () => {
  const source = fs.readFileSync(path.join(root,'../README.md'),'utf8');
  for (const page of JSON.parse(fs.readFileSync(path.join(root,'content-map.json'))))
    assert.equal(fs.readFileSync(path.join(root,`docs/${page.id}.md`),'utf8'), renderPage(source,page));
});
