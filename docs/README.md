# VascX documentation website

From the repository root, start the local development server:

```bash
cd docs
nvm install
npm ci
npm start
```

`nvm install` selects the Node version pinned in `.nvmrc` (skip this command if
that version is already available through another Node manager). Run `npm ci`
once after cloning, and again when dependencies change.

Open http://localhost:3000/retinalysis-vascx/. The development server reloads
when site files change; stop it with Ctrl+C. Edit the Markdown files in `docs/`
directly; starting or building the site does not overwrite them.
To use a different port, run `npm start -- --port 3001`.

To preview the production build instead, run these commands from `docs/`:

```bash
npm run build
npm run serve
```

This site builds independently from its Markdown pages and committed public
assets. Python, Rotterdam data, and sibling projects are not required.
`npm run content:validate` checks generated asset hashes; authored pages do not
need to match the repository README.

`sidebars.js` is the final authority for navigation, including the two top-level
sections **Biomarkers** and **Test-retest reliability**. It lists pages explicitly;
the Python generator never edits it.

- Edit About, usage, output and biomarker pages directly in `docs/`.
  These are manually maintained pages, independent of the repository README. `$...$` and `$$...$$` render equations using
  the site's bundled KaTeX styles and fonts.
- `docs_vascx` generates `docs/reliability/<id>.mdx`, containing evaluated
  combinations and ICC results. Add `reliability/<id>` to `sidebars.js` to show a
  new report in the navigation.
- Edit each reliability page's introduction in
  `src/content/reliability/macula-centered.mdx` or
  `src/content/reliability/od-centered.mdx`. The generated pages import these
  manually maintained files above the metadata table; regeneration never writes
  them. They support Markdown and MDX and are not standalone navigation pages.
- The same generator creates high-quality JPEG copies of report plots, at most
  2048 pixels along the longer side, in `static/assets/biomarkers/<id>/`.
  Original report images remain unchanged. `plots.json` in each directory inventories the
  assets and their measurement columns. Authored pages reference them using paths
  such as `/assets/biomarkers/macula-centered/cre_arteries.jpg`.

Regenerate reliability data and report images in the external `docs_vascx`
project. Files recorded in `generated-assets.json` are generated; author pages
outside that manifest. Only assets referenced by current pages or source components
are published; unused generated files are removed on regeneration. Use complete
literal `/assets/...` URLs in components so the publisher can discover them.
Site builds require only the committed public files.

Validation:

```bash
npm test
npx playwright install chromium
npm run test:browser
```

The Documentation workflow builds pull requests and deploys successful `main`
builds to https://eyened.github.io/retinalysis-vascx/ using GitHub Pages Actions.
For first deployment, set the repository's Pages source to **GitHub Actions**.
The `github-pages` environment must allow deployment from `main`.

The **Test-retest reliability** category opens a generated overview with the study
methods, device image counts and paired examples. The two result pages retain only
the evaluated version and feature set in their metadata tables. To regenerate all
of these, run the external `docs_vascx` runner with `--stage icc`; the paired-example
seed is configured in its `config.yaml`.

Biomarker and reliability images use the shared `src/theme/MDXComponents/Img`
modal. Click or keyboard-activate a figure to open it fitted to the window,
and use Escape or Close to return to the page.
The **Outputs → Analysis and QC** page is authored directly.
