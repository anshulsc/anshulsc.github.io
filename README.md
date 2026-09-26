# anshulsc.github.io

Personal academic portfolio of **Anshul Singh** — publications, projects, course
notes, and a photo gallery. Fully static, hand-authored HTML/CSS/JS with no
build step; hosted on **GitHub Pages** (custom domain in `CNAME`), so pushing to
`main` is the deploy.

## Layout

- `index.html`, `books.html`, `gallery.html` — top-level pages
- `css/` — shared styles; homepage CSS is mostly inline in `index.html`
- `js/` — site scripts (theme, homepage ornament, notes gate)
- `images/`, `data/` — imagery and served PDFs (CV, posters, slides)
- `writing/` — the writing section (`writing/index.html` is its index):
  `writing/series/` holds the per-course note series (`deep-gen/`,
  `reinforce-llms/`, `cs294-158/`, `reading/`), `writing/posts/` the
  standalone posts (`fl/`, `ddp/`, `iit-research/`, `dic-research/`), all
  sharing the skin in `writing/shared/`

Draft/lecture sources (`writing/series/*/draft-notes/`, `my_notes/`, `raw/`,
`waterloo-ml/`) are
gitignored — local working material, deliberately not published.

## Preview locally

```
python3 -m http.server 8000   # then open http://localhost:8000/
```

Contributor/agent docs live in `CLAUDE.md` and `writing/shared/WORKFLOW.md`.
