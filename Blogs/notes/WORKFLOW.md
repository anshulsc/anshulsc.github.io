# Lecture Notes Workflow

How a lecture becomes a chapter. Written 18 August 2026; the visual version is
at https://claude.ai/code/artifact/02984ade-0a85-4c84-bb50-13293409476a

The executable version is the **`lecture-notes` skill** (`~/.claude/skills/`),
and the voice rules it defers to live in
`.claude/skills/new-dgm-chapter/references/writing-style.md` — that file is
canonical for prose; this one is the shape of the process.

This governs every course series under `Blogs/` — `deep-gen` and
`reinforce_LLMs` were built this way before it was written down, and are the
reference for what "done" looks like.

---

## 1. One note, two speeds

Quick recall and depth are different documents pretending to be one, so every
chapter carries both layers. **The fast layer is written last** — a summary
drafted alongside the chapter is a summary of what you hoped the lecture said.

| | Layer 1 — the recap | Layer 2 — the chapter |
|---|---|---|
| Time | 90 seconds | ~20 minutes |
| Holds | 5–8 one-sentence claims · the 3 formulas worth memorising · one diagram · what it unlocks next | derivations with their plain-words twin · figures · widgets · real code |
| Lives | top of the chapter, **and** the body of the `note` node in OSCAR | the chapter page |

The recap living in OSCAR is the point: a recap should never require opening
the blog.

## 2. Handing a lecture over

Paste into chat, the moment the lecture is finished:

```
course: dist-train
lecture: 6
---
<raw scrape — transcript, bullet vomit, half-sentences, whatever exists.
Rough is expected.>
```

- `course` — a slug from §5.
- `lecture` — number in the course, or a video timestamp if that is all that is known.
- Anything else is optional and useful. "I never got the weight-absorption bit"
  changes what gets written; confusion is the best material in a note.

Alternatives: name a file `dist-train-lec-06.md` and give its path (no header
needed), or leave several in `Blogs/_inbox/` and say *process the inbox* — they
are worked in order, one commit each.

## 3. The loop

1. **The dump is committed untouched** to `<series>/raw/lec-NN.md`. It is the
   source and is never rewritten, so every later claim stays auditable against
   what the lecture actually said.
2. **The chapter is built** — the eight passes in §4.
3. **The recap is compressed** out of the finished chapter.
4. **A preview goes back before anything ships**: rendered screenshot, section
   list, figures and widgets by name, measured reading time.
5. **Publish** — commit and push; live in about a minute. Series index card and
   `sitemap.xml` update in the same commit (the repo's rule for any new page).
6. **OSCAR catches up** — a `note` node holding the recap as its body, linked
   `part_of` the course node, and `lecturesDone` ticks by one.

## 4. The eight passes

1. **Skeleton** — 6–12 numbered `## N.` sections. Structural, not cosmetic:
   `notes-core.js` builds the TOC from `h2` alone, so the section list *is* the
   chapter's argument.
2. **Rewrite** — plain words before every derivation, first person, honest about
   what was confusing. Student doubts from the lecture are **woven in as the
   author's own uncertainty**, never labelled, never a Q&A block — but every one
   of them must be answerable from the chapter alone.
3. **Math** — KaTeX `$…$` / `$$…$$`. Every symbol defined at first use, a symbol
   table past six new ones, display blocks that scroll rather than overflow.
4. **Figures — the original first.** Hunt down what the lecturer or the paper
   actually drew before anything is invented: the course's own slides, site or
   repo first (the original *and* already in the course's symbols), then the
   paper, then the author's blog or talk, then official docs. Vendor it where
   the licence permits — local copy under the series' `assets/`, never a
   hotlink, `ATTRIBUTION.md` beside it. Where it does not, and arXiv's default
   licence is the common case on a public site, redraw it faithfully with the
   original in front of you and caption *"after Figure N of [paper]"*. Invent a
   figure only where no original exists. Every course keeps its own `assets/`
   and its own notation — write the notation table first and take the labels
   from it.
5. **Widgets** — one to three, only where interaction teaches what a picture
   cannot. Canvas 2D, no dependencies, `prefers-reduced-motion` respected.
6. **Code** — real excerpts from the course's own repository, trimmed to the
   teaching core. If it does not run, it does not go in.
7. **The recap** — Layer 1. A claim that cannot survive one sentence means the
   section it came from is not finished.
8. **Wire and check** — head meta, measured reading time, prev/next, index card,
   sitemap. Then the preview.

## 5. Series registry

| Slug | Course | Lectures | Notes |
|---|---|---|---|
| `dgm` | Deep Generative Models | 11 | live — 11 chapters |
| `rl-llms` | Reinforcement Learning for LLMs | 7 | live — 7 chapters |
| `dist-train` | Distributed Training from First Principles (Umar Jamil) | 30 chapters | planned, ~12 notes |
| `cs285` | Deep Reinforcement Learning · Berkeley | 25 | on request |
| `cs294-158` | Deep Unsupervised Learning · Berkeley | 13 | on request |
| `ai-safety` | AI Safety · Tübingen | 14 | on request |
| `dl-spec` | Deep Learning Specialization | 125 | out of scope — covered everywhere |
| `math-ml` | Mathematics Foundations for ML | 70 | out of scope — reference, not notes |

A new course needs one entry in `series.json` and a folder; the chapter list is
derived from the video's own chapters or playlist.

**Sizing: one lecture = one chapter, however long.** This is the established
rule and it overrides tidiness — grouping lectures silently reorders the course
and breaks the "re-reading it should feel like sitting through it again" test.
For a course that is one long video, the video's own chapters are the lectures,
so `dist-train` is 30 notes rather than the 12 first proposed.

## 6. Layout

```
Blogs/notes/            ← shared by every course
├─ WORKFLOW.md          this file
├─ plain.css            the house skin — see §6.1
├─ notes-core.css       older card skin, still used by nothing
├─ notes-core.js        markdown render, section anchors, copy buttons
└─ series.json          slug → title, accent, OSCAR course node, chapters

Blogs/<series>/         ← one folder per course
├─ index.html           the link tree
├─ ch-01.html …
├─ raw/lec-01.md        the dumps, untouched
├─ widgets.js           this course's demos
└─ assets/svg/          figures
```

### 6.1 The skin

The notes are personal, so as of 19 August 2026 they are styled to read like a
notebook rather than a publication — the model is
<https://victorlecomte.com/notes/>. `plain.css` is the whole of it and every new
series uses it: set `<body class="plain">`, load `../notes/plain.css`, and load
nothing else of the old stylesheets. Fira Sans 300, plain `#2a7ae2` links, one
780px column. No cards, no borders, no background shading, no topic tags, no
hero, no reading times, no sidebar TOC, no prev/next.

What survives survives because it teaches: figures, widgets, KaTeX, code, and
the `§10` / `Ch.9, §10` cross-reference popovers. Widgets are the one element
allowed a frame; name the mount `<div class="note-widget" id="widget-x">`
(`.dgm-widget` is the older alias and is styled identically).

**The index is a link tree, not a syllabus.** `<h2>` per theme, then a flat
`<ul>` of chapter links with an optional `<span class="gloss">`. An entry with
no link and `class="todo"` is a note not yet written — the gaps are deliberately
part of the map, which is how you can see what the series is still missing.

`deep-gen` and `cs294-158` are both on it. **`reinforce_LLMs` is deliberately
not** — it keeps `rl-notes-style.css` and its original look.

One trap worth knowing: the `#sec-N` anchors that every cross-reference targets
are generated in JS, and used to be created inside `initTOC`, which returns
early when there is no `.toc-list`. Dropping the sidebar therefore broke every
`§N` link silently. Both `notes-core.js` and `dgm-notes.js` now build them in a
separate `initSectionAnchors()`; keep it that way.

**Outstanding housekeeping:** the move to `plain.css` left three stylesheets
with no page pointing at them — `deep-gen/dgm-notes-style.css` (1,116 lines),
`cs294-158/cs294-notes-style.css`, and `notes/notes-core.css`. They are kept for
now only as the way back to the card design; delete them once the plain skin has
proved itself. `css/theme.css` likewise still carries dark rules for `.chapter-card`,
`.series-hero` and `.toc-*`, which nothing renders any more.
`reinforce_LLMs/rl-notes-style.css` (805 lines) stays live and is deliberately
excluded from all of this.

## 7. Done means

- The recap alone would carry a conversation about the lecture.
- It opens with an italic paragraph on why this lecture mattered.
- No symbol undefined; no formula without its plain-words twin.
- At least one figure, and one widget or worked micro-example.
- It closes with the thing that would otherwise be forgotten in a month.
- Reading time measured, not guessed.
- Holds at 390 px; math scrolls instead of breaking the page.
