/**
 * RL Notes Series — Shared JavaScript
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
                var url = 'ch-' + paddedCh + '.html';
                return '<a href="' + url + '" class="chapter-link">' + match + '</a>';
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

    // ── Initialize Everything ──

    // ── Inline the figure SVGs ──
    // Swapping <img src="x.svg"> for the SVG markup itself puts the figure
    // inside the page's stylesheet, so the --fig-* tokens in plain.css reach
    // it and the drawing follows the light/dark theme. The SVGs keep their
    // original colours as var() fallbacks, so if this never runs (file://,
    // fetch blocked, JS off) the <img> simply stays where it is.
    function inlineFigureSVGs() {
        var imgs = document.querySelectorAll('figure img[src$=".svg"], figure img[src*=".svg?"]');
        Array.prototype.forEach.call(imgs, function (img) {
            var src = img.getAttribute('src');
            fetch(src).then(function (r) {
                return r.ok ? r.text() : null;
            }).then(function (txt) {
                if (!txt) return;
                var doc = new DOMParser().parseFromString(txt, 'image/svg+xml');
                if (doc.querySelector('parsererror')) return;
                var svg = doc.querySelector('svg');
                if (!svg) return;
                svg.removeAttribute('width');
                svg.removeAttribute('height');
                var st = img.getAttribute('style');
                if (st) svg.setAttribute('style', st);
                if (img.alt) {
                    svg.setAttribute('role', 'img');
                    svg.setAttribute('aria-label', img.alt);
                }
                if (img.parentNode) img.parentNode.replaceChild(svg, img);
            }).catch(function () { /* leave the <img> in place */ });
        });
    }

    function init() {
        renderMarkdown();
        inlineFigureSVGs();
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
    }

    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
