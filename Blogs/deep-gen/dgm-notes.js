/**
 * DGM Notes Series — Shared JavaScript
 * Handles: TOC generation, sticky TOC, reading progress, 
 * KaTeX rendering, code copy buttons, chapter navigation
 */

(function () {
    'use strict';

    // ── Reading Progress Bar ──
    function initProgressBar() {
        const bar = document.querySelector('.reading-progress');
        if (!bar) return;

        function updateProgress() {
            const scrollTop = window.scrollY;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
            bar.style.width = Math.min(progress, 100) + '%';
        }

        window.addEventListener('scroll', updateProgress, { passive: true });
        updateProgress();
    }

    // ── Table of Contents Generation ──
    function initTOC() {
        const content = document.querySelector('.chapter-content');
        const tocList = document.querySelector('.toc-list');
        if (!content || !tocList) return;

        const headings = content.querySelectorAll('h2');
        if (headings.length === 0) return;

        headings.forEach(function (h, i) {
            // Ensure heading has an ID
            if (!h.id) {
                h.id = 'section-' + i;
            }
            // a second id keyed to the section's own number, so "§10" can anchor to it
            var num = (h.textContent.match(/^\s*(\d+)\./) || [])[1];
            if (num && !document.getElementById('sec-' + num)) {
                var anchor = document.createElement('span');
                anchor.id = 'sec-' + num;
                anchor.style.cssText = 'position:absolute';
                h.insertBefore(anchor, h.firstChild);
            }

            var li = document.createElement('li');
            var a = document.createElement('a');
            a.href = '#' + h.id;
            a.textContent = h.textContent.replace(/^[\d]+\.\s*/, ''); // strip leading numbers
            a.addEventListener('click', function (e) {
                e.preventDefault();
                var target = document.getElementById(h.id);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    history.replaceState(null, '', '#' + h.id);
                }
            });
            li.appendChild(a);
            tocList.appendChild(li);
        });

        // Highlight active section on scroll
        var tocLinks = tocList.querySelectorAll('a');
        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    var id = entry.target.id;
                    tocLinks.forEach(function (link) {
                        link.classList.toggle('active', link.getAttribute('href') === '#' + id);
                    });
                }
            });
        }, {
            rootMargin: '-20px 0px -70% 0px',
            threshold: 0
        });

        headings.forEach(function (h) {
            observer.observe(h);
        });
    }

    // ── KaTeX Rendering ──
    function renderMath() {
        if (typeof katex === 'undefined' || typeof renderMathInElement === 'undefined') return;

        var content = document.querySelector('.chapter-content');
        if (!content) return;

        renderMathInElement(content, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\[', right: '\\]', display: true },
                { left: '\\(', right: '\\)', display: false }
            ],
            throwOnError: false,
            trust: true,
            macros: {
                '\\argmax': '\\operatorname{argmax}',
                '\\argmin': '\\operatorname{argmin}'
            }
        });
    }

    // ── Code Copy Buttons ──
    function initCopyButtons() {
        var pres = document.querySelectorAll('.chapter-content pre');

        pres.forEach(function (pre) {
            var btn = document.createElement('button');
            btn.className = 'code-copy-btn';
            btn.textContent = 'Copy';
            btn.addEventListener('click', function () {
                var code = pre.querySelector('code');
                var text = code ? code.textContent : pre.textContent;
                navigator.clipboard.writeText(text).then(function () {
                    btn.textContent = 'Copied!';
                    btn.classList.add('copied');
                    setTimeout(function () {
                        btn.textContent = 'Copy';
                        btn.classList.remove('copied');
                    }, 2000);
                });
            });
            pre.style.position = 'relative';
            pre.appendChild(btn);
        });
    }

    // ── Markdown Rendering with marked.js ──
    function renderMarkdown() {
        var rawEl = document.getElementById('raw-markdown');
        var target = document.getElementById('content-target');
        if (!rawEl || !target) return;

        var raw = rawEl.textContent || rawEl.innerHTML;

        if (typeof marked !== 'undefined') {
            var mathBlocks = [];
            
            // 1. Replace display math block $$ ... $$
            var temp = raw.replace(/\$\$([\s\S]+?)\$\$/g, function (match, equation) {
                var placeholder = '<!-- DISPLAY_MATH_' + mathBlocks.length + ' -->';
                mathBlocks.push({ placeholder: placeholder, content: '$$' + equation + '$$' });
                return placeholder;
            });

            // 2. Replace inline math block $ ... $ (single-line only to avoid cross-paragraph matching)
            temp = temp.replace(/\$([^\$\n]+?)\$/g, function (match, equation) {
                var placeholder = '<!-- INLINE_MATH_' + mathBlocks.length + ' -->';
                mathBlocks.push({ placeholder: placeholder, content: '$' + equation + '$' });
                return placeholder;
            });

            // 3. Render Markdown
            var html = '';
            if (typeof marked.parse === 'function') {
                html = marked.parse(temp);
            } else if (typeof marked === 'function') {
                html = marked(temp);
            } else {
                html = temp;
            }

            // Auto-convert chapter links: Ch.X, §Y or Ch.X
            html = html.replace(/\bCh\.(\d+)(?:,\s*§(\d+))?\b/g, function(match, chNum, secNum) {
                var paddedCh = chNum.padStart(2, '0');
                var url = 'ch-' + paddedCh + '.html' + (secNum ? '#sec-' + secNum : '');
                return '<a href="' + url + '" class="chapter-link" data-ch="' + paddedCh + '"' +
                    (secNum ? ' data-sec="' + secNum + '"' : '') + '>' + match + '</a>';
            });

            // 4. Restore math blocks using IIFE to avoid capture bugs, and escape HTML characters to prevent browser parsing errors
            for (var i = 0; i < mathBlocks.length; i++) {
                (function(item) {
                    var escapedContent = item.content
                        .replace(/&/g, '&amp;')
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;');
                    html = html.replace(item.placeholder, function() { return escapedContent; });
                })(mathBlocks[i]);
            }

            target.innerHTML = html;
        }
    }

    // ── Estimated Reading Time ──
    function calcReadingTime() {
        var el = document.getElementById('reading-time');
        var content = document.getElementById('raw-markdown');
        if (!el || !content) return;

        var text = content.textContent || content.innerHTML;
        var words = text.split(/\s+/).length;
        var minutes = Math.ceil(words / 200); // ~200 wpm for technical content
        el.textContent = minutes + ' min read';
    }

    // ── Hover Previews for Cross-References ──
    // "(6)" shows the tagged equation; "§10" the section; "Ch.9, §10" pulls the
    // section out of another chapter. The point is never having to leave the page.
    function initRefPreviews() {
        var content = document.querySelector('.chapter-content');
        if (!content) return;

        var pop = document.createElement('div');
        pop.className = 'ref-pop';
        pop.setAttribute('role', 'tooltip');
        document.body.appendChild(pop);

        // ---- index the tagged equations on this page ----
        var eqs = {};
        Array.prototype.forEach.call(content.querySelectorAll('.katex-display'), function (d) {
            var tag = d.querySelector('.tag');
            if (!tag) return;
            var n = (tag.textContent.match(/\d+/) || [])[0];
            if (n) eqs[n] = d;
        });

        // ---- index this page's sections ----
        var secs = {};
        Array.prototype.forEach.call(content.querySelectorAll('h2'), function (h) {
            var n = (h.textContent.match(/^\s*(\d+)\./) || [])[1];
            if (n) secs[n] = h;
        });

        // ---- linkify "(6)" and "§10" in prose, leaving rendered math untouched ----
        var eqKeys = Object.keys(eqs);
        var walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, null);
        var texts = [], node;
        while ((node = walker.nextNode())) {
            if (node.parentNode.closest('.katex, a, code, pre, .ref-pop')) continue;
            if (/\(\d{1,2}\)|§\d{1,2}/.test(node.nodeValue)) texts.push(node);
        }
        texts.forEach(function (t) {
            var html = t.nodeValue
                .replace(/\((\d{1,2})\)/g, function (m, n) {
                    return eqKeys.indexOf(n) === -1 ? m : '<a class="ref-link" data-eq="' + n + '">' + m + '</a>';
                })
                .replace(/§(\d{1,2})/g, function (m, n) {
                    return secs[n] ? '<a class="ref-link" data-secref="' + n + '">' + m + '</a>' : m;
                });
            if (html === t.nodeValue) return;
            var span = document.createElement('span');
            span.innerHTML = html;
            t.parentNode.replaceChild(span, t);
        });

        // ---- fetch + render a snippet from another chapter, cached ----
        var cache = {};
        function chapterMarkdown(ch) {
            if (cache[ch]) return cache[ch];
            cache[ch] = fetch('ch-' + ch + '.html')
                .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
                .then(function (txt) {
                    var m = txt.match(/<script id="raw-markdown" type="text\/template">([\s\S]*?)<\/script>/);
                    return m ? m[1] : null;
                })
                .catch(function () { return null; });
            return cache[ch];
        }

        function renderSnippet(md, target) {
            // same pipeline the page itself uses, so math and emphasis survive
            var blocks = [];
            var temp = md.replace(/\$\$([\s\S]+?)\$\$/g, function (mm, eq) {
                var ph = '<!-- M' + blocks.length + ' -->';
                blocks.push({ ph: ph, body: '$$' + eq + '$$' });
                return ph;
            }).replace(/\$([^\$\n]+?)\$/g, function (mm, eq) {
                var ph = '<!-- M' + blocks.length + ' -->';
                blocks.push({ ph: ph, body: '$' + eq + '$' });
                return ph;
            });
            var out = (typeof marked !== 'undefined' && marked.parse) ? marked.parse(temp) : temp;
            blocks.forEach(function (b) {
                out = out.replace(b.ph, function () {
                    return b.body.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                });
            });
            target.innerHTML = out;
            if (typeof renderMathInElement !== 'undefined') {
                renderMathInElement(target, {
                    delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
                    throwOnError: false, trust: true,
                    macros: { '\\argmax': '\\operatorname{argmax}', '\\argmin': '\\operatorname{argmin}' }
                });
            }
        }

        function sectionSlice(md, n) {
            var re = new RegExp('^##\\s+' + n + '\\.\\s+(.*)$', 'm');
            var m = re.exec(md);
            if (!m) return null;
            var rest = md.slice(m.index + m[0].length);
            var end = rest.search(/\n##\s/);
            var body = (end === -1 ? rest : rest.slice(0, end)).trim();
            // keep it to a preview: first couple of paragraphs, and drop widget mounts
            body = body.replace(/<div class="dgm-widget"[^>]*><\/div>/g, '')
                       .replace(/<figure>[\s\S]*?<\/figure>/g, '');
            var paras = body.split(/\n\s*\n/).filter(function (x) { return x.trim(); }).slice(0, 3);
            return { title: m[1], body: paras.join('\n\n') };
        }

        // ---- show / hide ----
        var hideTimer = null;
        function place(el) {
            var r = el.getBoundingClientRect();
            pop.style.visibility = 'hidden';
            pop.classList.add('on');
            var pw = pop.offsetWidth, ph = pop.offsetHeight;
            var left = Math.min(Math.max(8, r.left + r.width / 2 - pw / 2), window.innerWidth - pw - 8);
            var top = r.bottom + 10;
            if (top + ph > window.innerHeight - 8) top = Math.max(8, r.top - ph - 10);
            pop.style.left = left + 'px';
            pop.style.top = top + 'px';
            pop.style.visibility = 'visible';
        }

        function show(el, build) {
            clearTimeout(hideTimer);
            pop.innerHTML = '';
            build(function (label, node) {
                pop.innerHTML = '';
                if (label) {
                    var h = document.createElement('div');
                    h.className = 'ref-pop-label';
                    h.textContent = label;
                    pop.appendChild(h);
                }
                pop.appendChild(node);
                place(el);
            });
            place(el);
        }
        function hide() { hideTimer = setTimeout(function () { pop.classList.remove('on'); }, 140); }

        pop.addEventListener('mouseenter', function () { clearTimeout(hideTimer); });
        pop.addEventListener('mouseleave', hide);

        function attach(el, build) {
            el.addEventListener('mouseenter', function () { show(el, build); });
            el.addEventListener('mouseleave', hide);
            el.addEventListener('focus', function () { show(el, build); });
            el.addEventListener('blur', hide);
            if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '0');
        }

        // equation references
        Array.prototype.forEach.call(content.querySelectorAll('[data-eq]'), function (el) {
            attach(el, function (done) {
                var src = eqs[el.getAttribute('data-eq')];
                var clone = src.cloneNode(true);
                var wrap = document.createElement('div');
                wrap.className = 'ref-pop-body';
                wrap.appendChild(clone);
                done('equation (' + el.getAttribute('data-eq') + ') — this page', wrap);
            });
        });

        // same-page section references
        Array.prototype.forEach.call(content.querySelectorAll('[data-secref]'), function (el) {
            attach(el, function (done) {
                var h = secs[el.getAttribute('data-secref')];
                var wrap = document.createElement('div');
                wrap.className = 'ref-pop-body';
                var n = h.nextElementSibling, added = 0;
                while (n && n.tagName !== 'H2' && added < 2) {
                    if (n.tagName === 'P' || n.tagName === 'UL') { wrap.appendChild(n.cloneNode(true)); added++; }
                    n = n.nextElementSibling;
                }
                done(h.textContent.trim() + ' — this page', wrap);
            });
        });

        // cross-chapter references
        Array.prototype.forEach.call(content.querySelectorAll('a.chapter-link[data-ch]'), function (el) {
            attach(el, function (done) {
                var ch = el.getAttribute('data-ch'), sec = el.getAttribute('data-sec');
                var wrap = document.createElement('div');
                wrap.className = 'ref-pop-body';
                wrap.textContent = 'Loading…';
                done('Chapter ' + parseInt(ch, 10) + (sec ? ', §' + sec : ''), wrap);
                chapterMarkdown(ch).then(function (md) {
                    if (!md) { wrap.textContent = 'Preview unavailable — open the chapter directly.'; return; }
                    if (sec) {
                        var sl = sectionSlice(md, sec);
                        if (!sl) { wrap.textContent = 'Section not found.'; return; }
                        renderSnippet(sl.body, wrap);
                        var lab = pop.querySelector('.ref-pop-label');
                        if (lab) lab.textContent = 'Ch.' + parseInt(ch, 10) + ', §' + sec + ' — ' + sl.title;
                    } else {
                        var first = md.split(/\n##\s/)[0].replace(/^\s*\*[\s\S]*?\*\s*$/m, '').trim();
                        renderSnippet(first.split(/\n\s*\n/).slice(0, 2).join('\n\n'), wrap);
                    }
                    place(el);
                });
            });
        });
    }

    // ── Initialize Everything ──
    function init() {
        renderMarkdown();
        calcReadingTime();

        // Highlight code blocks
        if (typeof hljs !== 'undefined') {
            hljs.highlightAll();
        }

        // Render math (after markdown is in DOM)
        renderMath();

        // Interactive features
        initProgressBar();
        initTOC();
        initCopyButtons();
        initRefPreviews();
    }

    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
