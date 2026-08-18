# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Static personal academic website (Anshul Singh) served via **GitHub Pages** at the domain in `CNAME`. There is **no build step, no framework, and no package manager** — every page is hand-authored HTML/CSS/JS committed directly to the repo, and deployment happens automatically when changes land on `main`.

## Development & "build"

- **No build, lint, or test tooling.** To preview, serve the repo root over HTTP so relative asset paths and CDN scripts resolve:
  ```
  python3 -m http.server 8000   # then visit http://localhost:8000/
  ```
- **Deploy = push to `main`.** GitHub Pages publishes the repo root; any committed HTML/CSS/JS/asset change goes live.
- Third-party libraries (KaTeX, marked.js, highlight.js, three.js, Font Awesome, Academicons, Google Fonts) load from **CDNs at runtime** — nothing is vendored. Match the exact CDN URLs already used in sibling pages when adding a dependency.

## Structure & how pages relate

- **`index.html`** — main single-page CV/homepage. Large and mostly self-contained (most CSS inline in a `<style>` block); also pulls top-level `stylesheet.css`.
- **`books.html`, `gallery.html`** — standalone pages. Gallery images are **hardcoded `<img>` tags** pointing at `images/gallery/*.jpg` (no data-driven loop); add a photo by adding both the file and the markup.
- **`Blogs/`** — the writing section, its own mini-site keyed off `Blogs/blog-index.html` and styled by `Blogs/research-style.css`. Two content patterns coexist:
  1. **Markdown-in-HTML blogs** (`ddp.html`, `fl.html`, …): the `.md` source is embedded in a hidden `#raw-markdown` element, rendered client-side with **marked.js** into `#content-target`, then syntax-highlighted with **highlight.js**. These pages string-rewrite image paths (e.g. `/assets/ddp/` → `ddp/`) before rendering — keep image folders next to the HTML and match that convention.
  2. **Pure hand-written HTML articles** (`iit-research.html`, `dic-research.html`).
- **`Blogs/reinforce_LLMs/`** — self-contained multi-chapter "RL Notes" series (`index.html` + `ch-01`…`ch-07.html`) sharing `rl-notes-style.css` and `rl-notes.js`. `rl-notes.js` auto-generates the table of contents from `<h2>` headings and drives the reading-progress bar, active-section highlighting, code-copy buttons, and chapter nav. Math is **KaTeX** (auto-render). When adding a chapter, mirror an existing chapter's `<head>` (same fonts + stylesheets) and section structure so TOC/nav wire up automatically.
- **`notes/`** — additional standalone HTML notes (e.g. `activation_steering/`).
- **`Blogs/notes/WORKFLOW.md`** — how a watched lecture becomes a chapter: the intake format, the two note layers (90-second recap + full chapter), the eight refinement passes, the series registry and slugs. Read it before starting or extending any course-notes series.
- **`js/mobius.js`** — "The Loop": a dependency-free canvas-2D Möbius-strip ornament (depth-sorted shaded surface slices, contour edges, flowing particle glints) fixed bottom-right on the homepage via `#visual-canvas`. Its accent color follows scroll position through the section palette (purple/blue/orange/grey) and its rotation tracks reading progress; clicking it unrolls the strip into a flat ribbon while scrolling to top, then re-forms; respects `prefers-reduced-motion`, pauses on hidden tabs, hidden ≤768px. (`js/blackhole.js` is an alternate black-hole ornament for the same mount; `js/visuals.js` is an older three.js experiment that was never wired in — only one ornament script should be loaded.)
- **`data/`** — served PDFs (CV, papers). **`images/`** — all imagery, including `images/books_img/` and `images/gallery/`.

## SEO / housekeeping

`sitemap.xml`, `robots.txt`, and the `google*.html` verification file are manually maintained. When adding or renaming a top-level page or blog, update `sitemap.xml` (and its `lastmod` dates) to match.

## Conventions

- Styling mixes inline `<style>` blocks with shared stylesheets — check whether a page already has inline styles before adding a separate CSS rule.
- Newer `Blogs/` pages use Crimson Pro / Inter / JetBrains Mono via Google Fonts; the homepage uses its own font stack. Keep a page visually consistent with its section, not the whole repo.
- Use relative asset paths (`../index.html`, `images/…`) so pages work both on GitHub Pages and when opened locally.
