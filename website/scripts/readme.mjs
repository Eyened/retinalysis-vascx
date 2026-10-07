/** Slice README blocks without rewriting their prose or code. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { unified } from 'unified';
import remarkParse from 'remark-parse';

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const parse = source => unified().use(remarkParse).parse(source);
const textContent = node => node.value ?? (node.children ?? []).map(textContent).join('');

export function sections(source) {
  const nodes = parse(source).children.filter(node => node.type === 'heading' && node.depth === 2);
  const result = new Map();
  nodes.forEach((node, index) => {
    const title = textContent(node);
    if (result.has(title)) throw new Error(`Duplicate README heading: ${title}`);
    result.set(title, source.slice(node.position.end.offset, nodes[index + 1]?.position.start.offset).trim());
  });
  return result;
}

export function selectBlock(sourceSections, selector) {
  let body = sourceSections.get(selector.section);
  if (body === undefined) throw new Error(`Missing README section: ${selector.section}`);
  for (const key of ['from', 'until']) {
    if (!selector[key]) continue;
    const position = body.indexOf(selector[key]);
    if (position < 0 || body.indexOf(selector[key], position + 1) >= 0)
      throw new Error(`Missing or ambiguous README boundary: ${selector[key]}`);
    body = key === 'from' ? body.slice(position) : body.slice(0, position);
  }
  return body.trim();
}

export function relocateLinks(source) {
  const replacements = [];
  function walk(node) {
    if ((node.type === 'link' || node.type === 'image') && node.url.startsWith('./')) {
      const start = node.position.start.offset;
      const original = source.slice(start, node.position.end.offset);
      const offset = original.lastIndexOf(node.url);
      replacements.push({ start: start + offset, end: start + offset + node.url.length,
        value: 'https://github.com/Eyened/retinalysis-vascx/blob/main/' + node.url.slice(2) });
    }
    for (const child of node.children ?? []) walk(child);
  }
  walk(parse(source));
  for (const edit of replacements.sort((a, b) => b.start - a.start))
    source = source.slice(0, edit.start) + edit.value + source.slice(edit.end);
  return source;
}

export function renderPage(source, page) {
  const sourceSections = sections(source);
  const blocks = page.blocks.map(selector => {
    const body = selectBlock(sourceSections, selector);
    const heading = selector.from || selector.until || page.blocks.length === 1 ? '' : `## ${selector.section}\n\n`;
    return heading + body;
  });
  const slug = page.id === 'about' ? 'slug: /\n' : '';
  return `---\ntitle: ${JSON.stringify(page.title)}\n${slug}---\n\n` + relocateLinks(blocks.join('\n\n')) + '\n';
}

export function generate() {
  const source = fs.readFileSync(path.join(root, '../README.md'), 'utf8');
  const pages = JSON.parse(fs.readFileSync(path.join(root, 'content-map.json')));
  const fingerprints = {};
  for (const page of pages) {
    if (!/^[a-z0-9/-]+$/.test(page.id) || page.id.includes('..')) throw new Error('Unsafe page ID');
    const relative = `docs/${page.id}.md`;
    const destination = path.join(root, relative);
    const content = renderPage(source, page);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.writeFileSync(destination, content);
    fingerprints[relative] = createHash('sha256').update(content).digest('hex');
  }
  fs.writeFileSync(path.join(root, 'readme-provenance.json'), JSON.stringify({
    source_sha256: createHash('sha256').update(source).digest('hex'), pages: fingerprints,
  }, null, 2) + '\n');
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) generate();
