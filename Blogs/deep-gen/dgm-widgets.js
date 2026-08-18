/**
 * DGM Notes Series — Interactive Widgets
 * Each widget mounts into an empty <div class="dgm-widget" id="widget-..."> that
 * lives inside the rendered markdown. Loaded after dgm-notes.js so the content
 * is already in the DOM.
 */

(function () {
    'use strict';

    var TEAL = '#0e7490';
    var TEAL_FILL = 'rgba(14,116,144,0.14)';
    var ORANGE = '#d97706';
    var ORANGE_FILL = 'rgba(217,119,6,0.12)';

    function byId(id) { return document.getElementById(id); }

    function make(tag, className, parent, text) {
        var e = document.createElement(tag);
        if (className) e.className = className;
        if (text !== undefined) e.textContent = text;
        if (parent) parent.appendChild(e);
        return e;
    }

    function makeTitle(root, text) {
        make('div', 'w-title', root, text);
    }

    function makeCanvas(root, h) {
        var c = make('canvas', null, root);
        var w = Math.min(root.clientWidth - 4, 640) || 620;
        var dpr = window.devicePixelRatio || 1;
        c.width = w * dpr;
        c.height = h * dpr;
        c.style.width = w + 'px';
        c.style.height = h + 'px';
        var ctx = c.getContext('2d');
        ctx.scale(dpr, dpr);
        c._w = w;
        c._h = h;
        return c;
    }

    function slider(parent, labelText, min, max, step, value) {
        var wrap = make('div', 'w-slider-row', parent);
        var lab = make('label', null, wrap, labelText + ' ');
        var input = make('input', null, wrap);
        input.type = 'range';
        input.min = min; input.max = max; input.step = step; input.value = value;
        var val = make('span', 'w-slider-val', wrap, value);
        input._val = val;
        return input;
    }

    // ─────────────────────────────────────────────────────────
    // Widget 1 (Lecture 1): Run the random experiment
    // ─────────────────────────────────────────────────────────
    function initExperiment() {
        var root = byId('widget-experiment');
        if (!root) return;
        makeTitle(root, 'Try it — run the random experiment yourself');

        var faces = ['⚀', '⚁', '⚂', '⚃', '⚄', '⚅'];
        var probs = [0.05, 0.10, 0.15, 0.20, 0.22, 0.28]; // a loaded die
        var counts = [0, 0, 0, 0, 0, 0];
        var n = 0;
        var reveal = false;

        var controls = make('div', null, root);
        var b1 = make('button', null, controls, 'Roll once');
        var b200 = make('button', null, controls, 'Roll 200 times');
        var bReset = make('button', 'secondary', controls, 'Reset');
        var bReveal = make('button', 'secondary', controls, 'Reveal true distribution');

        var status = make('div', 'w-readout', root, 'ω = ?   →   X(ω) = ?   (n = 0 trials)');
        var canvas = makeCanvas(root, 210);
        var ctx = canvas.getContext('2d');

        make('div', 'w-note', root,
            'The die is secretly loaded. Each roll is one run of the random experiment: it produces an outcome ω ∈ Ω, ' +
            'and the fixed function X maps it to a number. The bars are the empirical distribution of X; keep rolling and it ' +
            'converges to the induced distribution Pₓ (outlined) — the thing every dataset is a finite glimpse of.');

        function sampleFace() {
            var r = Math.random(), cum = 0;
            for (var i = 0; i < 6; i++) { cum += probs[i]; if (r < cum) return i; }
            return 5;
        }

        function draw() {
            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var pad = 34, baseY = h - 30;
            var bw = (w - 2 * pad) / 6;
            var maxP = 0.4;
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(pad - 8, baseY); ctx.lineTo(w - pad + 8, baseY); ctx.stroke();

            for (var i = 0; i < 6; i++) {
                var x = pad + i * bw + bw * 0.18;
                var barW = bw * 0.64;
                if (n > 0) {
                    var f = counts[i] / n;
                    var bh = Math.min(f / maxP, 1) * (baseY - 28);
                    ctx.fillStyle = TEAL_FILL;
                    ctx.strokeStyle = TEAL;
                    ctx.lineWidth = 1.5;
                    ctx.fillRect(x, baseY - bh, barW, bh);
                    ctx.strokeRect(x, baseY - bh, barW, bh);
                }
                if (reveal) {
                    var th = Math.min(probs[i] / maxP, 1) * (baseY - 28);
                    ctx.strokeStyle = ORANGE;
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([5, 3]);
                    ctx.strokeRect(x - 2, baseY - th, barW + 4, th);
                    ctx.setLineDash([]);
                }
                ctx.fillStyle = '#555';
                ctx.font = '16px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(faces[i], x + barW / 2, baseY + 18);
                ctx.font = '10px Inter, sans-serif';
                ctx.fillStyle = '#999';
                ctx.fillText('X=' + (i + 1), x + barW / 2, baseY + 29);
            }
            if (reveal) {
                ctx.fillStyle = ORANGE;
                ctx.font = '11px Inter, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText('dashed = true Pₓ', w - 10, 16);
            }
        }

        function roll(times) {
            var last = 0;
            for (var t = 0; t < times; t++) { last = sampleFace(); counts[last]++; n++; }
            status.textContent = 'ω = ' + faces[last] + '   →   X(ω) = ' + (last + 1) + '   (n = ' + n + ' trials)';
            draw();
        }

        b1.addEventListener('click', function () { roll(1); });
        b200.addEventListener('click', function () { roll(200); });
        bReset.addEventListener('click', function () {
            counts = [0, 0, 0, 0, 0, 0]; n = 0;
            status.textContent = 'ω = ?   →   X(ω) = ?   (n = 0 trials)';
            draw();
        });
        bReveal.addEventListener('click', function () {
            reveal = !reveal;
            bReveal.textContent = reveal ? 'Hide true distribution' : 'Reveal true distribution';
            draw();
        });

        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 2 (Lecture 2): σ-algebra checker
    // ─────────────────────────────────────────────────────────
    function initSigma() {
        var root = byId('widget-sigma');
        if (!root) return;
        makeTitle(root, 'Try it — build a σ-algebra on Ω = {1, 2, 3, 4}');

        var FULL = 15; // 1111
        var selected = {};

        function popcount(m) { var c = 0; while (m) { c += m & 1; m >>= 1; } return c; }
        function label(mask) {
            if (mask === 0) return '∅';
            if (mask === FULL) return 'Ω';
            var parts = [];
            for (var i = 0; i < 4; i++) if (mask & (1 << i)) parts.push(i + 1);
            return '{' + parts.join(',') + '}';
        }

        var presetRow = make('div', null, root);
        make('label', null, presetRow, 'Presets:  ');
        var presets = [
            { name: 'Trivial', masks: [0, FULL] },
            { name: 'Even / odd', masks: [0, 10, 5, FULL] },            // {2,4}=1010=10, {1,3}=0101=5
            { name: 'σ({1})', masks: [0, 1, 14, FULL] },
            { name: 'Power set', masks: null },
            { name: 'Clear', masks: [] }
        ];

        var chipBox = make('div', 'w-chipbox', root);
        var masksSorted = [];
        for (var m = 0; m <= FULL; m++) masksSorted.push(m);
        masksSorted.sort(function (a, b) { return popcount(a) - popcount(b) || a - b; });

        var chips = {};
        masksSorted.forEach(function (mask) {
            var chip = make('span', 'chip', chipBox, label(mask));
            chip.addEventListener('click', function () {
                selected[mask] = !selected[mask];
                chip.classList.toggle('on', !!selected[mask]);
                check();
            });
            chips[mask] = chip;
        });

        var verdict = make('div', 'verdict', root, '');
        make('div', 'w-note', root,
            'Click subsets to include them in your collection ℱ, or start from a preset. The checker applies the three ' +
            'axioms live: Ω must be in ℱ, complements must stay in ℱ, and unions must stay in ℱ. ' +
            'Note how few of the 65,536 possible collections survive — and how "even/odd" is a perfectly legal σ-algebra ' +
            'that simply refuses to answer fine-grained questions.');

        presets.forEach(function (p) {
            var b = make('button', p.name === 'Clear' ? 'secondary' : null, presetRow, p.name);
            b.addEventListener('click', function () {
                selected = {};
                var list = p.masks === null ? masksSorted : p.masks;
                list.forEach(function (mask) { selected[mask] = true; });
                masksSorted.forEach(function (mask) {
                    chips[mask].classList.toggle('on', !!selected[mask]);
                });
                check();
            });
        });

        function check() {
            var members = [];
            for (var m = 0; m <= FULL; m++) if (selected[m]) members.push(m);

            if (members.length === 0) {
                verdict.textContent = 'Empty collection — pick some events.';
                verdict.style.color = '#888';
                return;
            }
            var has = function (m) { return !!selected[m]; };
            var problem = null;

            if (!has(FULL)) problem = 'Ω itself is missing — the certain event must always be askable.';
            if (!problem) {
                for (var i = 0; i < members.length && !problem; i++) {
                    var c = FULL ^ members[i];
                    if (!has(c)) problem = 'complement of ' + label(members[i]) + ' (= ' + label(c) + ') is missing.';
                }
            }
            if (!problem) {
                for (var a = 0; a < members.length && !problem; a++) {
                    for (var b = a + 1; b < members.length && !problem; b++) {
                        var u = members[a] | members[b];
                        if (!has(u)) problem = label(members[a]) + ' ∪ ' + label(members[b]) + ' = ' + label(u) + ' is missing.';
                    }
                }
            }

            if (problem) {
                verdict.textContent = '✗ Not a σ-algebra: ' + problem;
                verdict.style.color = '#b91c1c';
            } else {
                verdict.textContent = '✓ This is a σ-algebra with ' + members.length + ' events.';
                verdict.style.color = '#0e7490';
            }
        }

        // start on the even/odd preset — the interesting one
        [0, 10, 5, FULL].forEach(function (mask) { selected[mask] = true; chips[mask].classList.add('on'); });
        check();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 3 (Lecture 2): PDF vs CDF
    // ─────────────────────────────────────────────────────────
    function erf(x) {
        var s = x < 0 ? -1 : 1;
        x = Math.abs(x);
        var t = 1 / (1 + 0.3275911 * x);
        var y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
        return s * y;
    }
    function gaussPdf(x, mu, s) { return Math.exp(-0.5 * Math.pow((x - mu) / s, 2)) / (s * Math.sqrt(2 * Math.PI)); }
    function gaussCdf(x, mu, s) { return 0.5 * (1 + erf((x - mu) / (s * Math.SQRT2))); }

    function initPdfCdf() {
        var root = byId('widget-pdfcdf');
        if (!root) return;
        makeTitle(root, 'Try it — density vs. distribution');

        var sMu = slider(root, 'μ', -3, 3, 0.1, 0);
        var sSig = slider(root, 'σ', 0.2, 2.5, 0.05, 1);
        var sA = slider(root, 'a', -5, 5, 0.1, -1);
        var sB = slider(root, 'b', -5, 5, 0.1, 1);

        var canvas = makeCanvas(root, 230);
        var ctx = canvas.getContext('2d');
        var readout = make('div', 'w-readout', root, '');

        make('div', 'w-note', root,
            'Solid teal: the density pₓ (PDF). Dashed orange: the distribution Pₓ (CDF), on a 0–1 scale. The shaded ' +
            'area between a and b is the actual probability — exactly the rise of the CDF over [a, b]. Now drag σ below ' +
            '0.5 and watch the density peak climb above 1: densities are not probabilities until you integrate them.');

        function draw() {
            var mu = +sMu.value, sig = +sSig.value;
            var a = Math.min(+sA.value, +sB.value), b = Math.max(+sA.value, +sB.value);
            sMu._val.textContent = mu.toFixed(1);
            sSig._val.textContent = sig.toFixed(2);
            sA._val.textContent = a.toFixed(1);
            sB._val.textContent = b.toFixed(1);

            var w = canvas._w, h = canvas._h;
            var padL = 36, padR = 36, baseY = h - 26, topY = 14;
            var x0 = -5, x1 = 5;
            var maxPdf = 2.1; // pdf axis scale

            function px(x) { return padL + (x - x0) / (x1 - x0) * (w - padL - padR); }
            function pyPdf(v) { return baseY - Math.min(v / maxPdf, 1) * (baseY - topY); }
            function pyCdf(v) { return baseY - v * (baseY - topY); }

            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();

            // shaded probability area
            ctx.beginPath();
            ctx.moveTo(px(a), baseY);
            for (var x = a; x <= b + 1e-9; x += 0.02) ctx.lineTo(px(x), pyPdf(gaussPdf(x, mu, sig)));
            ctx.lineTo(px(b), baseY);
            ctx.closePath();
            ctx.fillStyle = TEAL_FILL;
            ctx.fill();

            // pdf curve
            ctx.beginPath();
            for (x = x0; x <= x1; x += 0.02) {
                var y = pyPdf(gaussPdf(x, mu, sig));
                x === x0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2; ctx.stroke();

            // cdf curve
            ctx.beginPath();
            ctx.setLineDash([6, 4]);
            for (x = x0; x <= x1; x += 0.02) {
                y = pyCdf(gaussCdf(x, mu, sig));
                x === x0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 1.8; ctx.stroke();
            ctx.setLineDash([]);

            // a & b markers
            [a, b].forEach(function (v) {
                ctx.strokeStyle = '#aaa'; ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.beginPath(); ctx.moveTo(px(v), baseY); ctx.lineTo(px(v), topY); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = '#666'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
                ctx.fillText(v === a ? 'a' : 'b', px(v), baseY + 14);
            });

            // axis hints
            ctx.fillStyle = TEAL; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('pₓ (density)', padL + 2, 12);
            ctx.fillStyle = ORANGE; ctx.textAlign = 'right';
            ctx.fillText('Pₓ (CDF, 0–1)', w - padR - 2, 12);

            var prob = gaussCdf(b, mu, sig) - gaussCdf(a, mu, sig);
            var peak = gaussPdf(mu, mu, sig);
            readout.textContent = 'P(a ≤ X ≤ b) = ' + prob.toFixed(3) +
                '    |    peak density pₓ(μ) = ' + peak.toFixed(2) + (peak > 1 ? '  ← bigger than 1!' : '');
        }

        [sMu, sSig, sA, sB].forEach(function (s) { s.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 4 (Lecture 2): fit a Gaussian to a bimodal truth
    // ─────────────────────────────────────────────────────────
    function initFit() {
        var root = byId('widget-fit');
        if (!root) return;
        makeTitle(root, 'Try it — divergence minimization, live');

        function truePdf(x) {
            return 0.5 * gaussPdf(x, -2, 0.7) + 0.5 * gaussPdf(x, 1.8, 0.55);
        }

        var sMu = slider(root, 'μ of model Q', -4, 4, 0.05, 0.5);
        var sSig = slider(root, 'σ of model Q', 0.3, 3, 0.05, 0.6);

        var controls = make('div', null, root);
        var bFit = make('button', null, controls, 'Auto-fit (gradient descent)');
        var bReset = make('button', 'secondary', controls, 'Reset');

        var canvas = makeCanvas(root, 220);
        var ctx = canvas.getContext('2d');
        var readout = make('div', 'w-readout', root, '');

        make('div', 'w-note', root,
            'Orange: the true data density pₓ — bimodal, like real data (MNIST has ten modes, not one). Teal: your model ' +
            'p_θ, a single Gaussian with θ = {μ, σ}. Slide θ around, or let gradient descent minimize ' +
            'Dᴋʟ(pₓ ‖ p_θ) for you. It never reaches zero: the best a unimodal family can do is stretch across ' +
            'both peaks. This is exactly why we swap the Gaussian for a neural network g_θ(z).');

        var animId = null;

        function kl(mu, sig) {
            var sum = 0, dx = 0.02;
            for (var x = -6; x <= 6; x += dx) {
                var p = truePdf(x);
                if (p < 1e-12) continue;
                var q = Math.max(gaussPdf(x, mu, sig), 1e-12);
                sum += p * Math.log(p / q) * dx;
            }
            return sum;
        }

        function draw() {
            var mu = +sMu.value, sig = +sSig.value;
            sMu._val.textContent = mu.toFixed(2);
            sSig._val.textContent = sig.toFixed(2);

            var w = canvas._w, h = canvas._h;
            var padL = 20, padR = 20, baseY = h - 22, topY = 12;
            var x0 = -5.5, x1 = 5.5, maxV = 0.45;

            function px(x) { return padL + (x - x0) / (x1 - x0) * (w - padL - padR); }
            function py(v) { return baseY - Math.min(v / maxV, 1) * (baseY - topY); }

            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();

            // true density (filled orange)
            ctx.beginPath();
            ctx.moveTo(px(x0), baseY);
            for (var x = x0; x <= x1; x += 0.03) ctx.lineTo(px(x), py(truePdf(x)));
            ctx.lineTo(px(x1), baseY);
            ctx.closePath();
            ctx.fillStyle = ORANGE_FILL; ctx.fill();
            ctx.beginPath();
            for (x = x0; x <= x1; x += 0.03) {
                var y = py(truePdf(x));
                x === x0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 1.8; ctx.stroke();

            // model (teal)
            ctx.beginPath();
            for (x = x0; x <= x1; x += 0.03) {
                y = py(gaussPdf(x, mu, sig));
                x === x0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.2; ctx.stroke();

            ctx.font = '11px Inter, sans-serif';
            ctx.fillStyle = ORANGE; ctx.textAlign = 'left';
            ctx.fillText('true pₓ (unknown to the model)', padL + 4, 12);
            ctx.fillStyle = TEAL; ctx.textAlign = 'right';
            ctx.fillText('model p_θ = N(μ, σ)', w - padR - 4, 12);

            readout.textContent = 'D_KL(pₓ ‖ p_θ) = ' + kl(mu, sig).toFixed(4) + ' nats';
        }

        function stopAnim() {
            if (animId) { cancelAnimationFrame(animId); animId = null; bFit.textContent = 'Auto-fit (gradient descent)'; }
        }

        bFit.addEventListener('click', function () {
            if (animId) { stopAnim(); return; }
            bFit.textContent = 'Stop';
            var steps = 0;
            function step() {
                var mu = +sMu.value, sig = +sSig.value;
                var hFd = 0.01;
                var gMu = (kl(mu + hFd, sig) - kl(mu - hFd, sig)) / (2 * hFd);
                var gSig = (kl(mu, sig + hFd) - kl(mu, sig - hFd)) / (2 * hFd);
                mu -= 0.08 * gMu;
                sig -= 0.04 * gSig;
                sig = Math.max(0.3, Math.min(3, sig));
                mu = Math.max(-4, Math.min(4, mu));
                sMu.value = mu;
                sSig.value = sig;
                draw();
                steps++;
                if (steps < 260 && (Math.abs(gMu) > 1e-3 || Math.abs(gSig) > 1e-3)) {
                    animId = requestAnimationFrame(step);
                } else {
                    stopAnim();
                }
            }
            animId = requestAnimationFrame(step);
        });

        bReset.addEventListener('click', function () {
            stopAnim();
            sMu.value = 0.5; sSig.value = 0.6;
            draw();
        });

        [sMu, sSig].forEach(function (s) { s.addEventListener('input', function () { stopAnim(); draw(); }); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 5 (Lecture 2): entropy, cross-entropy, KL
    // ─────────────────────────────────────────────────────────
    function initKl() {
        var root = byId('widget-kl');
        if (!root) return;
        makeTitle(root, 'Try it — entropy, cross-entropy & KL');

        var P = [0.4, 0.3, 0.2, 0.1];
        var cats = ['x₁', 'x₂', 'x₃', 'x₄'];

        var sliders = [];
        for (var i = 0; i < 4; i++) {
            sliders.push(slider(root, 'Q(' + cats[i] + ')', 1, 100, 1, 25));
        }
        var controls = make('div', null, root);
        var bMatch = make('button', null, controls, 'Set Q = P');
        var bUnif = make('button', 'secondary', controls, 'Uniform Q');

        var canvas = makeCanvas(root, 190);
        var ctx = canvas.getContext('2d');
        var readout = make('div', 'w-readout', root, '');
        var readout2 = make('div', 'w-readout', root, '');

        make('div', 'w-note', root,
            'P (solid teal) is the true distribution; Q (dashed orange) is your model’s belief — the sliders are ' +
            'auto-normalized. Watch the gap H(P,Q) − H(P): it is exactly Dᴋʟ(P ‖ Q), hits zero only when the ' +
            'bars coincide, and never goes negative. The second line shows Dᴋʟ(Q ‖ P) — almost always a ' +
            'different number. That asymmetry is why KL is a divergence, not a distance.');

        function getQ() {
            var raw = sliders.map(function (s) { return +s.value; });
            var sum = raw.reduce(function (a, b) { return a + b; }, 0);
            return raw.map(function (v) { return v / sum; });
        }

        function H(p) {
            var s = 0;
            for (var i = 0; i < p.length; i++) if (p[i] > 0) s -= p[i] * Math.log2(p[i]);
            return s;
        }
        function crossH(p, q) {
            var s = 0;
            for (var i = 0; i < p.length; i++) if (p[i] > 0) s -= p[i] * Math.log2(Math.max(q[i], 1e-12));
            return s;
        }

        function draw() {
            var Q = getQ();
            sliders.forEach(function (s, i) { s._val.textContent = Q[i].toFixed(2); });

            var w = canvas._w, h = canvas._h;
            var pad = 36, baseY = h - 26;
            var bw = (w - 2 * pad) / 4;
            var maxV = 0.75;

            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(pad - 8, baseY); ctx.lineTo(w - pad + 8, baseY); ctx.stroke();

            for (var i = 0; i < 4; i++) {
                var xc = pad + i * bw;
                var pW = bw * 0.30, qW = bw * 0.30;
                var pH = P[i] / maxV * (baseY - 24);
                var qH = Q[i] / maxV * (baseY - 24);
                // P bar
                ctx.fillStyle = TEAL_FILL; ctx.strokeStyle = TEAL; ctx.lineWidth = 1.5;
                ctx.fillRect(xc + bw * 0.12, baseY - pH, pW, pH);
                ctx.strokeRect(xc + bw * 0.12, baseY - pH, pW, pH);
                // Q bar
                ctx.strokeStyle = ORANGE; ctx.setLineDash([5, 3]);
                ctx.strokeRect(xc + bw * 0.52, baseY - qH, qW, qH);
                ctx.setLineDash([]);
                ctx.fillStyle = ORANGE_FILL;
                ctx.fillRect(xc + bw * 0.52, baseY - qH, qW, qH);

                ctx.fillStyle = '#666'; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'center';
                ctx.fillText(cats[i], xc + bw / 2, baseY + 16);
            }
            ctx.fillStyle = TEAL; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('P (true)', pad, 12);
            ctx.fillStyle = ORANGE; ctx.textAlign = 'right';
            ctx.fillText('Q (model)', w - pad, 12);

            var hp = H(P), hpq = crossH(P, Q), hqp = crossH(Q, P), hq = H(Q);
            readout.textContent = 'H(P) = ' + hp.toFixed(3) + '   H(P,Q) = ' + hpq.toFixed(3) +
                '   D_KL(P‖Q) = ' + (hpq - hp).toFixed(3) + ' bits';
            readout2.textContent = 'D_KL(Q‖P) = ' + (hqp - hq).toFixed(3) + ' bits  ← not the same!';
        }

        sliders.forEach(function (s) { s.addEventListener('input', draw); });
        bMatch.addEventListener('click', function () {
            sliders.forEach(function (s, i) { s.value = Math.round(P[i] * 100); });
            draw();
        });
        bUnif.addEventListener('click', function () {
            sliders.forEach(function (s) { s.value = 25; });
            draw();
        });

        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 6 (Lecture 3): a deterministic function warps a distribution
    // ─────────────────────────────────────────────────────────
    function initWarp() {
        var root = byId('widget-warp');
        if (!root) return;
        makeTitle(root, 'Try it — a fixed function reshapes randomness');

        var gens = [
            { name: 'g(z) = z²', fn: function (z) { return z * z; } },
            { name: 'g(z) = √z', fn: function (z) { return Math.sqrt(z); } },
            { name: 'g(z) = z³', fn: function (z) { return z * z * z; } }
        ];
        var current = 0;

        var controls = make('div', null, root);
        var btns = gens.map(function (g, i) {
            var b = make('button', i === 0 ? null : 'secondary', controls, g.name);
            b.addEventListener('click', function () {
                current = i;
                btns.forEach(function (bb, j) { bb.className = j === i ? '' : 'secondary'; });
                draw();
            });
            return b;
        });
        var bResample = make('button', 'secondary', controls, 'Resample');

        var canvas = makeCanvas(root, 210);
        var ctx = canvas.getContext('2d');

        make('div', 'w-note', root,
            'Left: 3,000 samples of z ~ U[0,1] — flat, boring, easy to generate. Right: the very same samples pushed ' +
            'through a fixed deterministic function g. Nothing random happens inside g, yet the output distribution is ' +
            'completely different — squaring drags mass toward 0, square-rooting drags it toward 1. A deep network ' +
            'g_θ does exactly this, just with millions of knobs: it warps easy noise into the distribution of real data.');

        var samples = [];
        function resample() {
            samples = [];
            for (var i = 0; i < 3000; i++) samples.push(Math.random());
        }

        function hist(values, bins) {
            var h = new Array(bins).fill(0);
            values.forEach(function (v) {
                var b = Math.min(bins - 1, Math.floor(v * bins));
                h[b]++;
            });
            return h;
        }

        function drawHist(h, xoff, width, label, color, fill) {
            var w = canvas._w, hgt = canvas._h;
            var baseY = hgt - 34, topY = 22;
            var maxC = Math.max.apply(null, h) * 1.1;
            var bw = width / h.length;
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(xoff, baseY); ctx.lineTo(xoff + width, baseY); ctx.stroke();
            h.forEach(function (c, i) {
                var bh = c / maxC * (baseY - topY);
                ctx.fillStyle = fill;
                ctx.fillRect(xoff + i * bw, baseY - bh, bw - 1, bh);
            });
            ctx.fillStyle = color; ctx.font = '11.5px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText(label, xoff + width / 2, 14);
            ctx.fillStyle = '#999'; ctx.font = '10px Inter, sans-serif';
            ctx.fillText('0', xoff, baseY + 14);
            ctx.fillText('1', xoff + width, baseY + 14);
        }

        function draw() {
            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var colW = (w - 90) / 2;
            drawHist(hist(samples, 30), 20, colW, 'input z ~ U[0,1]', '#666', 'rgba(120,120,120,0.35)');
            var out = samples.map(gens[current].fn);
            drawHist(hist(out, 30), colW + 70, colW, 'output ' + gens[current].name, TEAL, TEAL_FILL);
            // arrow between
            ctx.strokeStyle = TEAL; ctx.lineWidth = 1.6;
            var midX = colW + 45, midY = h / 2 - 8;
            ctx.beginPath(); ctx.moveTo(midX - 14, midY); ctx.lineTo(midX + 10, midY); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(midX + 4, midY - 5); ctx.lineTo(midX + 12, midY); ctx.lineTo(midX + 4, midY + 5); ctx.stroke();
        }

        bResample.addEventListener('click', function () { resample(); draw(); });
        resample();
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 7 (Lecture 3): Legendre–Fenchel conjugate explorer
    // f(u) = u ln u ; f*(t) = e^(t-1) ; u* = e^(t-1)
    // ─────────────────────────────────────────────────────────
    function initConjugate() {
        var root = byId('widget-conjugate');
        if (!root) return;
        makeTitle(root, 'Try it — the convex conjugate, drawn');

        function f(u) { return u <= 0 ? 0 : u * Math.log(u); }

        var sT = slider(root, 'slope t', -1.5, 2, 0.05, 0.6);
        var controls = make('div', null, root);
        var bEnv = make('button', 'secondary', controls, 'Show tangent-line envelope');
        var showEnv = false;

        var canvas = makeCanvas(root, 260);
        var ctx = canvas.getContext('2d');
        var readout = make('div', 'w-readout', root, '');

        make('div', 'w-note', root,
            'Teal curve: f(u) = u ln u. Grey dashed line: y = t·u, a line through the origin whose slope you control. ' +
            'The orange segment is the biggest vertical gap t·u − f(u) — that maximum gap IS the conjugate value f*(t), ' +
            'and it always occurs where the line is parallel to the curve. Shift the line down by f*(t) (solid orange) and ' +
            'it kisses the curve from below: a supporting line. Toggle the envelope to see the duality: drawing that ' +
            'supporting line for every t rebuilds f(u) perfectly — a convex curve is completely described by its tangent ' +
            'lines. That is the identity f(u) = sup_t {t·u − f*(t)}.');

        var U0 = 0.02, U1 = 3.1, Ymin = -0.6, Ymax = 3.4;

        function draw() {
            var t = +sT.value;
            sT._val.textContent = t.toFixed(2);
            var uStar = Math.exp(t - 1);          // argmax of tu - f(u)
            var fStar = uStar * t - f(uStar);      // = e^(t-1)

            var w = canvas._w, h = canvas._h;
            var padL = 34, padR = 16, padT = 10, padB = 26;
            function px(u) { return padL + (u - U0) / (U1 - U0) * (w - padL - padR); }
            function py(y) { return h - padB - (y - Ymin) / (Ymax - Ymin) * (h - padT - padB); }

            ctx.clearRect(0, 0, w, h);
            // axes
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, py(0)); ctx.lineTo(w - padR, py(0)); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(px(U0), padT); ctx.lineTo(px(U0), h - padB); ctx.stroke();
            ctx.fillStyle = '#888'; ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'center';
            [1, 2, 3].forEach(function (u) {
                ctx.fillText(u, px(u), py(0) + 14);
                ctx.beginPath(); ctx.moveTo(px(u), py(0) - 3); ctx.lineTo(px(u), py(0) + 3); ctx.stroke();
            });
            ctx.fillText('u', w - padR - 6, py(0) - 8);

            // envelope of supporting lines
            if (showEnv) {
                ctx.lineWidth = 1;
                for (var tt = -1.4; tt <= 2.01; tt += 0.2) {
                    var fs = Math.exp(tt - 1);
                    ctx.strokeStyle = 'rgba(217,119,6,0.25)';
                    ctx.beginPath();
                    ctx.moveTo(px(U0), py(tt * U0 - fs));
                    ctx.lineTo(px(U1), py(tt * U1 - fs));
                    ctx.stroke();
                }
            }

            // line y = t u (through origin)
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1.4; ctx.setLineDash([6, 4]);
            ctx.beginPath(); ctx.moveTo(px(U0), py(t * U0)); ctx.lineTo(px(U1), py(t * U1)); ctx.stroke();
            ctx.setLineDash([]);

            // supporting line y = t u - f*(t)
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 1.8;
            ctx.beginPath(); ctx.moveTo(px(U0), py(t * U0 - fStar)); ctx.lineTo(px(U1), py(t * U1 - fStar)); ctx.stroke();

            // f(u)
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.4;
            ctx.beginPath();
            for (var u = U0; u <= U1; u += 0.01) {
                var y = py(f(u));
                u === U0 ? ctx.moveTo(px(u), y) : ctx.lineTo(px(u), y);
            }
            ctx.stroke();

            // the max-gap segment at u*
            if (uStar > U0 && uStar < U1) {
                ctx.strokeStyle = ORANGE; ctx.lineWidth = 2.6;
                ctx.beginPath(); ctx.moveTo(px(uStar), py(t * uStar)); ctx.lineTo(px(uStar), py(f(uStar))); ctx.stroke();
                ctx.fillStyle = ORANGE;
                ctx.beginPath(); ctx.arc(px(uStar), py(f(uStar)), 3.5, 0, 7); ctx.fill();
                ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left';
                ctx.fillText('gap = f*(t)', px(uStar) + 8, py((t * uStar + f(uStar)) / 2));
            }

            // labels
            ctx.fillStyle = TEAL; ctx.font = '11.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('f(u) = u ln u', px(2.35), py(f(2.6)));
            ctx.fillStyle = '#888';
            ctx.fillText('y = t·u', px(U1) - 44, py(t * U1) - 8);

            readout.textContent = 't = ' + t.toFixed(2) + '   →   best u* = ' + uStar.toFixed(2) +
                '   f*(t) = max gap = ' + fStar.toFixed(3);
        }

        bEnv.addEventListener('click', function () {
            showEnv = !showEnv;
            bEnv.textContent = showEnv ? 'Hide tangent-line envelope' : 'Show tangent-line envelope';
            draw();
        });
        sT.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 8 (Lecture 3): the f-divergence family
    // ─────────────────────────────────────────────────────────
    function initFdiv() {
        var root = byId('widget-fdiv');
        if (!root) return;
        makeTitle(root, 'Try it — one formula, many divergences');

        var fams = [
            { name: 'KL:  f(u) = u ln u', f: function (u) { return u <= 0 ? 0 : u * Math.log(u); } },
            { name: 'TV:  f(u) = ½|u−1|', f: function (u) { return 0.5 * Math.abs(u - 1); } },
            {
                name: 'JS:  f(u) = ½(u ln u − (u+1) ln((u+1)/2))',
                f: function (u) {
                    if (u <= 0) return 0.5 * Math.log(2);
                    return 0.5 * (u * Math.log(u) - (u + 1) * Math.log((u + 1) / 2));
                }
            }
        ];
        var current = 0;

        var controls = make('div', null, root);
        var btns = fams.map(function (fam, i) {
            var b = make('button', i === 0 ? null : 'secondary', controls, fam.name.split(':')[0]);
            b.addEventListener('click', function () {
                current = i;
                btns.forEach(function (bb, j) { bb.className = j === i ? '' : 'secondary'; });
                draw();
            });
            return b;
        });

        var sMu = slider(root, 'μ of Q', -3, 3, 0.05, 1.2);
        var sSig = slider(root, 'σ of Q', 0.4, 2.5, 0.05, 1);

        var canvas = makeCanvas(root, 220);
        var ctx = canvas.getContext('2d');
        var readout = make('div', 'w-readout', root, '');

        make('div', 'w-note', root,
            'Left: the chosen generator function f(u) — always convex, always zero at u = 1 (the black dot), because ' +
            'u = 1 means "the densities agree here" and agreement must cost nothing. Right: P = N(0,1) fixed (teal), ' +
            'Q = N(μ,σ) yours to move (orange), with D_f computed numerically. Slide Q onto P: every member of the ' +
            'family hits zero together. Slide Q far away: KL explodes, JS saturates at ln 2 ≈ 0.69, TV caps at 1. ' +
            'Same recipe, different penalty profiles — that choice shapes what a GAN learns.');

        function draw() {
            var mu = +sMu.value, sig = +sSig.value;
            sMu._val.textContent = mu.toFixed(2);
            sSig._val.textContent = sig.toFixed(2);
            var fam = fams[current];

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var colW = (w - 80) / 2;

            // ── left: f(u) plot ──
            var padL = 30, baseY = h - 30, topY = 20;
            var U1 = 3, Fmin = -0.45, Fmax = 2.2;
            function pxL(u) { return padL + u / U1 * (colW - padL); }
            function pyL(y) { return baseY - (y - Fmin) / (Fmax - Fmin) * (baseY - topY); }
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, pyL(0)); ctx.lineTo(colW, pyL(0)); ctx.stroke();
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.2;
            ctx.beginPath();
            for (var u = 0.02; u <= U1; u += 0.01) {
                var y = pyL(fam.f(u));
                u <= 0.03 ? ctx.moveTo(pxL(u), y) : ctx.lineTo(pxL(u), y);
            }
            ctx.stroke();
            ctx.fillStyle = '#111';
            ctx.beginPath(); ctx.arc(pxL(1), pyL(0), 4, 0, 7); ctx.fill();
            ctx.fillStyle = '#555'; ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('u=1, f=0', pxL(1), pyL(0) + 16);
            ctx.fillStyle = '#333'; ctx.font = '11px Inter, sans-serif';
            ctx.fillText(fam.name, colW / 2 + 10, 12);

            // ── right: P and Q ──
            var xoff = colW + 60;
            var X0 = -5, X1 = 5, maxV = 0.85;
            function pxR(x) { return xoff + (x - X0) / (X1 - X0) * (w - xoff - 14); }
            function pyR(v) { return baseY - Math.min(v / maxV, 1) * (baseY - topY); }
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(xoff, baseY); ctx.lineTo(w - 14, baseY); ctx.stroke();
            // P
            ctx.beginPath();
            for (var x = X0; x <= X1; x += 0.04) {
                var yy = pyR(gaussPdf(x, 0, 1));
                x === X0 ? ctx.moveTo(pxR(x), yy) : ctx.lineTo(pxR(x), yy);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2; ctx.stroke();
            // Q
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.04) {
                yy = pyR(gaussPdf(x, mu, sig));
                x === X0 ? ctx.moveTo(pxR(x), yy) : ctx.lineTo(pxR(x), yy);
            }
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = TEAL; ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('P = N(0,1)', xoff + 4, 12);
            ctx.fillStyle = ORANGE; ctx.textAlign = 'right';
            ctx.fillText('Q = N(μ,σ)', w - 16, 12);

            // numeric D_f = ∫ q f(p/q)
            var sum = 0, dx = 0.02;
            for (x = -8; x <= 8; x += dx) {
                var p = gaussPdf(x, 0, 1);
                var q = Math.max(gaussPdf(x, mu, sig), 1e-14);
                sum += q * fam.f(p / q) * dx;
            }
            readout.textContent = 'D_f(P ‖ Q) = ' + sum.toFixed(4) +
                (current === 1 ? '   (TV: capped at 1)' : current === 2 ? '   (JS: capped at ln 2 ≈ 0.693)' : '   (KL: unbounded)');
        }

        [sMu, sSig].forEach(function (s) { s.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 9 (Lecture 4): Jensen's inequality ⇒ D_f ≥ 0
    // ─────────────────────────────────────────────────────────
    function initJensen() {
        var root = byId('widget-jensen');
        if (!root) return;
        makeTitle(root, 'Try it — the proof that an f-divergence can never go negative');

        var fams = [
            { tag: 'KL', name: 'f(u) = u ln u', f: function (u) { return u <= 0 ? 0 : u * Math.log(u); } },
            { tag: 'TV', name: 'f(u) = ½|u − 1|', f: function (u) { return 0.5 * Math.abs(u - 1); } },
            { tag: 'χ²', name: 'f(u) = (u − 1)²', f: function (u) { return (u - 1) * (u - 1); } }
        ];
        var cur = 0;

        var controls = make('div', null, root);
        var btns = fams.map(function (fam, i) {
            var b = make('button', i === 0 ? null : 'secondary', controls, fam.tag);
            b.addEventListener('click', function () {
                cur = i;
                btns.forEach(function (bb, j) { bb.className = j === i ? '' : 'secondary'; });
                draw();
            });
            return b;
        });

        var sU1 = slider(root, 'ratio u₁ (a region where P_θ wins)', 0.05, 0.95, 0.01, 0.35);
        var sU2 = slider(root, 'ratio u₂ (a region where P_x wins)', 1.05, 3.5, 0.01, 2.4);
        var sLam = slider(root, 'weight λ on u₁', 0, 1, 0.01, 0.5);
        var bSnap = make('button', 'secondary', make('div', null, root), 'Snap so that E[u] = 1');

        var canvas = makeCanvas(root, 250);
        var ctx = canvas.getContext('2d');
        var readout = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'A divergence is an average of f evaluated at density ratios: D_f = E over P_θ of f(u), with u = pₓ/p_θ. ' +
            'Take just two ratios u₁, u₂ with weights λ and 1−λ. Because f is convex, the chord (grey) never dips below ' +
            'the curve (teal) — so the weighted average of the two f-values (grey dot) always sits at or above f of the ' +
            'weighted average ratio (teal dot). That is Jensen\'s inequality, and it holds for any number of points, hence ' +
            'for the whole integral. Now the punchline: the average ratio is pinned. E over P_θ of pₓ/p_θ = ∫p_θ·(pₓ/p_θ) = ' +
            '∫pₓ = 1, always. Press the snap button to put the teal dot exactly at u = 1, where every legal f is zero. ' +
            'So D_f ≥ f(1) = 0, with equality only when all the ratios collapse onto 1 — that is, when the two ' +
            'distributions are the same.');

        bSnap.addEventListener('click', function () {
            var u1 = +sU1.value, u2 = +sU2.value;
            sLam.value = ((1 - u2) / (u1 - u2)).toFixed(2);
            draw();
        });

        function draw() {
            var u1 = +sU1.value, u2 = +sU2.value, lam = +sLam.value;
            sU1._val.textContent = u1.toFixed(2);
            sU2._val.textContent = u2.toFixed(2);
            sLam._val.textContent = lam.toFixed(2);
            var f = fams[cur].f;

            var uBar = lam * u1 + (1 - lam) * u2;
            var fBar = lam * f(u1) + (1 - lam) * f(u2);
            var fOfBar = f(uBar);

            var w = canvas._w, h = canvas._h;
            var U0 = 0, U1 = 3.7;
            var lo = 0, hi = 0, uu;
            for (uu = 0.02; uu <= U1; uu += 0.05) { lo = Math.min(lo, f(uu)); hi = Math.max(hi, f(uu)); }
            var padY = (hi - lo) * 0.12;
            lo -= padY; hi += padY;

            var padL = 40, padR = 16, padT = 14, padB = 28;
            function px(u) { return padL + (u - U0) / (U1 - U0) * (w - padL - padR); }
            function py(y) { return h - padB - (y - lo) / (hi - lo) * (h - padT - padB); }

            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, py(0)); ctx.lineTo(w - padR, py(0)); ctx.stroke();
            ctx.strokeStyle = '#999';
            ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.stroke();

            // the u = 1 guide line
            ctx.strokeStyle = 'rgba(17,17,17,0.35)'; ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(px(1), padT); ctx.lineTo(px(1), h - padB); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#666'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('u = 1', px(1), h - padB + 14);
            [1, 2, 3].forEach(function (u) { if (u !== 1) ctx.fillText(u, px(u), h - padB + 14); });

            // curve
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.4;
            ctx.beginPath();
            for (uu = 0.02; uu <= U1; uu += 0.01) {
                var yy = py(f(uu));
                uu <= 0.03 ? ctx.moveTo(px(uu), yy) : ctx.lineTo(px(uu), yy);
            }
            ctx.stroke();

            // chord
            ctx.strokeStyle = '#888'; ctx.lineWidth = 1.6;
            ctx.beginPath(); ctx.moveTo(px(u1), py(f(u1))); ctx.lineTo(px(u2), py(f(u2))); ctx.stroke();
            ctx.fillStyle = '#888';
            [u1, u2].forEach(function (u) {
                ctx.beginPath(); ctx.arc(px(u), py(f(u)), 3.6, 0, 7); ctx.fill();
            });

            // the gap
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 2.6;
            ctx.beginPath(); ctx.moveTo(px(uBar), py(fBar)); ctx.lineTo(px(uBar), py(fOfBar)); ctx.stroke();
            ctx.fillStyle = '#555';
            ctx.beginPath(); ctx.arc(px(uBar), py(fBar), 4.2, 0, 7); ctx.fill();
            ctx.fillStyle = TEAL;
            ctx.beginPath(); ctx.arc(px(uBar), py(fOfBar), 4.2, 0, 7); ctx.fill();

            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = '#555';
            ctx.fillText('E[f(u)]  (average of the two f-values)', px(uBar) + 9, py(fBar) - 4);
            ctx.fillStyle = TEAL;
            ctx.fillText('f(E[u])', px(uBar) + 9, py(fOfBar) + 13);
            ctx.fillStyle = '#111';
            ctx.beginPath(); ctx.arc(px(1), py(0), 3.6, 0, 7); ctx.fill();

            readout.textContent =
                'E[u] = ' + uBar.toFixed(3) + '    E[f(u)] = ' + fBar.toFixed(4) +
                '    f(E[u]) = ' + fOfBar.toFixed(4) + '    gap = ' + (fBar - fOfBar).toFixed(4);

            if (Math.abs(uBar - 1) < 0.015) {
                verdict.textContent = '✓ E[u] = 1, so D_f ≈ ' + fBar.toFixed(4) + ' ≥ f(1) = 0 — the divergence cannot be negative.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = 'E[u] = ' + uBar.toFixed(2) + ' — for real densities this is forced to be exactly 1. Press snap.';
                verdict.style.color = '#999';
            }
        }

        [sU1, sU2, sLam].forEach(function (s) { s.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 10 (Lecture 4): the GAN's f and its conjugate
    // f(u) = u ln u − (u+1) ln(u+1) ;  f*(t) = −ln(1 − e^t), t < 0
    // ─────────────────────────────────────────────────────────
    function initGanConj() {
        var root = byId('widget-ganconj');
        if (!root) return;
        makeTitle(root, 'Try it — the GAN generator function and its conjugate');

        function fRaw(u) { return u <= 0 ? 0 : u * Math.log(u) - (u + 1) * Math.log(u + 1); }
        function fJs(u) { return u <= 0 ? Math.log(2) : u * Math.log(u) - (u + 1) * Math.log((u + 1) / 2); }
        function fStar(t) { return -Math.log(1 - Math.exp(t)); }

        var sT = slider(root, 'slope t  (must stay below 0)', -5, -0.05, 0.01, -1.2);

        var canvas = makeCanvas(root, 260);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');

        make('div', 'w-note', root,
            'Left: the f the vanilla GAN secretly uses (solid teal) and the properly normalised JS generator (dashed). ' +
            'They differ by the term (u+1)·ln 2, which is why the solid curve sits at −2 ln 2 ≈ −1.386 at u = 1 instead of 0 — ' +
            'the whole "the GAN loss is off by log 4" story is that vertical shift, and a shift has no gradient. ' +
            'Right: the conjugate f*(t) = −ln(1 − eᵗ). Drag t and watch it blow up as t approaches 0 from below: at t = 0 the ' +
            'argument 1 − eᵗ hits zero and the logarithm dies. That wall is the entire reason a sigmoid appears in GANs — ' +
            'the network\'s raw output would happily wander into the forbidden zone, so we route it through T = log σ(v), ' +
            'which is negative by construction. The readout also shows the identity that makes discriminators useful ' +
            'beyond GANs: at the optimum the density ratio is recoverable as pₓ/p_θ = D/(1 − D).');

        function draw() {
            var t = +sT.value;
            sT._val.textContent = t.toFixed(2);
            var D = Math.exp(t);
            var uStar = D / (1 - D);
            var fs = fStar(t);

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var gap = 44;
            var colW = (w - gap) / 2;
            var topY = 26, baseY = h - 34;

            // ── LEFT: f(u) ──
            var U0 = 0, U1 = 5, Fl = -2.2, Fh = 0.6;
            function pxL(u) { return 34 + (u - U0) / (U1 - U0) * (colW - 44); }
            function pyL(y) { return baseY - (y - Fl) / (Fh - Fl) * (baseY - topY); }
            ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(34, pyL(0)); ctx.lineTo(colW - 10, pyL(0)); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(34, topY); ctx.lineTo(34, baseY); ctx.stroke();

            ctx.strokeStyle = '#aaa'; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]);
            ctx.beginPath();
            for (var u = 0.02; u <= U1; u += 0.02) {
                var y2 = pyL(fJs(u));
                u <= 0.03 ? ctx.moveTo(pxL(u), y2) : ctx.lineTo(pxL(u), y2);
            }
            ctx.stroke(); ctx.setLineDash([]);

            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.4;
            ctx.beginPath();
            for (u = 0.02; u <= U1; u += 0.02) {
                var y = pyL(fRaw(u));
                u <= 0.03 ? ctx.moveTo(pxL(u), y) : ctx.lineTo(pxL(u), y);
            }
            ctx.stroke();

            // markers at u = 1
            ctx.strokeStyle = 'rgba(17,17,17,0.3)'; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(pxL(1), topY); ctx.lineTo(pxL(1), baseY); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#111';
            ctx.beginPath(); ctx.arc(pxL(1), pyL(0), 3.6, 0, 7); ctx.fill();
            ctx.fillStyle = TEAL;
            ctx.beginPath(); ctx.arc(pxL(1), pyL(fRaw(1)), 3.6, 0, 7); ctx.fill();

            // u* marker
            if (uStar > 0.02 && uStar < U1) {
                ctx.fillStyle = ORANGE;
                ctx.beginPath(); ctx.arc(pxL(uStar), pyL(fRaw(uStar)), 4.4, 0, 7); ctx.fill();
                ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
                ctx.fillText('u*', pxL(uStar), pyL(fRaw(uStar)) - 8);
            }

            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('f(u) = u ln u − (u+1) ln(u+1)', 34, 14);
            ctx.fillStyle = '#999'; ctx.fillText('JS version, f(1) = 0', 34, baseY + 28);
            ctx.fillStyle = '#666'; ctx.textAlign = 'center';
            ctx.fillText('u', colW - 14, pyL(0) + 14);
            ctx.fillText('−2 ln 2', pxL(1) + 30, pyL(fRaw(1)) + 4);

            // ── RIGHT: f*(t) ──
            var x0 = colW + gap;
            var T0 = -5, T1 = 0.6, Sl = -0.2, Sh = 3.2;
            function pxR(tt) { return x0 + (tt - T0) / (T1 - T0) * (w - x0 - 14); }
            function pyR(y) { return baseY - (y - Sl) / (Sh - Sl) * (baseY - topY); }
            ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x0, pyR(0)); ctx.lineTo(w - 14, pyR(0)); ctx.stroke();

            // forbidden zone t >= 0
            ctx.fillStyle = 'rgba(200,60,60,0.07)';
            ctx.fillRect(pxR(0), topY, pxR(T1) - pxR(0), baseY - topY);
            ctx.strokeStyle = '#c0392b'; ctx.lineWidth = 1.6;
            ctx.beginPath(); ctx.moveTo(pxR(0), topY); ctx.lineTo(pxR(0), baseY); ctx.stroke();
            ctx.fillStyle = '#c0392b'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('undefined', pxR(0.3), topY + 40);
            ctx.fillText('for t ≥ 0', pxR(0.3), topY + 53);

            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.4;
            ctx.beginPath();
            var started = false;
            for (var tt = T0; tt <= -0.005; tt += 0.005) {
                var v = fStar(tt);
                if (v > Sh) { started = false; continue; }
                var yv = pyR(v);
                started ? ctx.lineTo(pxR(tt), yv) : ctx.moveTo(pxR(tt), yv);
                started = true;
            }
            ctx.stroke();

            ctx.fillStyle = ORANGE;
            if (fs < Sh) {
                ctx.beginPath(); ctx.arc(pxR(t), pyR(fs), 4.4, 0, 7); ctx.fill();
                ctx.strokeStyle = 'rgba(217,119,6,0.5)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
                ctx.beginPath(); ctx.moveTo(pxR(t), pyR(0)); ctx.lineTo(pxR(t), pyR(fs)); ctx.stroke();
                ctx.setLineDash([]);
            }
            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('f*(t) = −ln(1 − eᵗ)', x0, 14);
            ctx.fillStyle = '#666'; ctx.textAlign = 'center';
            ctx.fillText('t', w - 20, pyR(0) + 14);

            r1.textContent = 't = ' + t.toFixed(2) + '    f*(t) = ' + fs.toFixed(3) +
                '    D = eᵗ = ' + D.toFixed(3);
            r2.textContent = 'implied density ratio  pₓ/p_θ = u* = D/(1−D) = ' + uStar.toFixed(3) +
                (uStar > 1 ? '   (real data is denser here)' : '   (the generator over-covers here)');
        }

        sT.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 11 (Lecture 4): the optimal discriminator, numerically
    // ─────────────────────────────────────────────────────────
    function initOptD() {
        var root = byId('widget-optd');
        if (!root) return;
        makeTitle(root, 'Try it — is D* = p/(p+q) really the maximiser?');

        var sMu = slider(root, 'μ of P_θ', -3, 3, 0.05, 1.3);
        var sSig = slider(root, 'σ of P_θ', 0.4, 2.5, 0.05, 0.8);
        var sS = slider(root, 'critic strength s', 0, 2, 0.02, 0.4);

        var canvas = makeCanvas(root, 250);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'P_x = N(0,1) is fixed (teal); P_θ = N(μ,σ) is yours to move (orange). The purple curve is the critic ' +
            'D_s(x) = σ(s · log(pₓ/p_θ)) — a whole family of discriminators indexed by one dial. At s = 0 it is the ' +
            'useless constant critic D ≡ ½; at s = 1 it is exactly the optimal D*(x) = pₓ/(pₓ + p_θ); past s = 1 it ' +
            'over-commits. Sweep s and watch the value V: it peaks at s = 1 and falls off on both sides, which is the ' +
            'calculus result made visible. Two more numbers worth staring at: at s = 0 the value is exactly −log 4, and ' +
            'at the peak it equals 2·D_JS − log 4. That is the whole "vanilla GAN = Jensen-Shannon, shifted" claim, ' +
            'checked numerically rather than taken on faith.');

        function stats(mu, sig, s) {
            var V = 0, js = 0, dx = 0.02;
            // wide enough that the tails of any Q the sliders can reach are fully captured,
            // otherwise V(D*) and 2·JS − log 4 stop agreeing in the 3rd decimal
            for (var x = -18; x <= 18; x += dx) {
                var p = gaussPdf(x, 0, 1);
                var q = Math.max(gaussPdf(x, mu, sig), 1e-300);
                var pp = Math.max(p, 1e-300);
                var lr = Math.log(pp / q);
                var D = 1 / (1 + Math.exp(-s * lr));
                D = Math.min(1 - 1e-12, Math.max(1e-12, D));
                V += (p * Math.log(D) + q * Math.log(1 - D)) * dx;
                var m = 0.5 * (pp + q);
                js += 0.5 * (pp * Math.log(pp / m) + q * Math.log(q / m)) * dx;
            }
            return { V: V, js: js };
        }

        function draw() {
            var mu = +sMu.value, sig = +sSig.value, s = +sS.value;
            sMu._val.textContent = mu.toFixed(2);
            sSig._val.textContent = sig.toFixed(2);
            sS._val.textContent = s.toFixed(2);

            var st = stats(mu, sig, s);
            var stOpt = stats(mu, sig, 1);

            var w = canvas._w, h = canvas._h;
            var padL = 34, padR = 40, topY = 22, baseY = h - 30;
            var X0 = -5.5, X1 = 5.5, maxV = 0.75;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (w - padL - padR); }
            function pyD(v) { return baseY - Math.min(v / maxV, 1) * (baseY - topY); }
            function pyProb(v) { return baseY - v * (baseY - topY); }

            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();

            // D = 0.5 reference
            ctx.strokeStyle = '#ddd'; ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(padL, pyProb(0.5)); ctx.lineTo(w - padR, pyProb(0.5)); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#bbb'; ctx.font = '9.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('D = ½', w - padR + 4, pyProb(0.5) + 3);
            ctx.fillText('D = 1', w - padR + 4, pyProb(1) + 3);
            ctx.fillText('D = 0', w - padR + 4, pyProb(0) + 3);

            // densities
            var x;
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.03) {
                var y = pyD(gaussPdf(x, 0, 1));
                x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2; ctx.stroke();
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.03) {
                y = pyD(gaussPdf(x, mu, sig));
                x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.stroke();
            ctx.setLineDash([]);

            // critic
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.02) {
                var p = Math.max(gaussPdf(x, 0, 1), 1e-300);
                var q = Math.max(gaussPdf(x, mu, sig), 1e-300);
                var D = 1 / (1 + Math.exp(-s * Math.log(p / q)));
                y = pyProb(D);
                x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2.4; ctx.stroke();

            ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('P_x = N(0,1)', padL + 2, 13);
            ctx.fillStyle = ORANGE; ctx.fillText('P_θ = N(μ,σ)', padL + 96, 13);
            ctx.fillStyle = '#7c3aed'; ctx.fillText('critic D_s(x)', padL + 196, 13);

            r1.textContent = 'V(D_s) = ' + st.V.toFixed(4) +
                '        V at s = 1  →  ' + stOpt.V.toFixed(4);
            r2.textContent = '2·D_JS − log 4 = ' + (2 * stOpt.js - Math.log(4)).toFixed(4) +
                '        −log 4 = ' + (-Math.log(4)).toFixed(4) +
                '        D_JS = ' + stOpt.js.toFixed(4);

            if (Math.abs(s - 1) < 0.02) {
                verdict.textContent = '✓ s = 1: the critic is exactly D* = pₓ/(pₓ+p_θ), and V equals 2·D_JS − log 4 to the last digit.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = 'V is ' + (stOpt.V - st.V).toFixed(4) + ' lower than at s = 1' +
                    (s < 1 ? ' — this critic is too timid.' : ' — this critic over-commits.');
                verdict.style.color = '#999';
            }
        }

        [sMu, sSig, sS].forEach(function (sl) { sl.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 12 (Lecture 4): a real GAN, trained live, in 1-D
    // ─────────────────────────────────────────────────────────
    function initGanTrain() {
        var root = byId('widget-gantrain');
        if (!root) return;
        makeTitle(root, 'Try it — train an actual GAN, right here');

        var TRUE_MU = 1.6, TRUE_SIG = 0.5;
        var LRD = 0.2, LRG = 0.15, BATCH = 128;

        var a, b, w0, w1, w2, steps, animId = null;
        var mode = 0; // 0 = minimax, 1 = non-saturating

        function reset() { a = 1.5; b = -1.6; w0 = 0; w1 = 0; w2 = 0; steps = 0; }
        reset();

        var controls = make('div', null, root);
        var bPlay = make('button', null, controls, '▶ Train');
        var b200 = make('button', 'secondary', controls, 'Run 200 steps');
        var bReset = make('button', 'secondary', controls, 'Reset');
        var bMode = make('button', 'secondary', controls, 'loss: minimax');

        var sK = slider(root, 'critic steps per generator step (k)', 1, 5, 1, 2);

        var canvas = makeCanvas(root, 260);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');

        make('div', 'w-note', root,
            'This is the real algorithm, not an animation of one. The generator is G_θ(z) = a·z + b with z ~ N(0,1), so ' +
            'P_θ = N(b, a) and the honest answer is a = 0.5, b = 1.6. The critic is a quadratic logit v(x) = w₀ + w₁x + w₂x², ' +
            'which is exactly rich enough to represent the optimal log-ratio for two Gaussians — a deliberately strong ' +
            'critic, per the "loose bound = useless gradient" argument. Each step draws a fresh minibatch of 128, runs k ' +
            'ascent steps on the critic, then one descent step on the generator, with gradients computed exactly as ' +
            'derived above. Things worth watching: the purple curve flattens toward ½ as the fakes become ' +
            'indistinguishable; the orange bump usually collapses to a spike before it widens out again (a one-dimensional ' +
            'preview of mode collapse); and even after convergence the parameters keep circling the answer instead of ' +
            'settling — that wobble is the saddle point, not a bug. Switch the generator loss to non-saturating and the ' +
            'early progress is visibly faster, because log(1−D) has almost no gradient while the critic is winning.');

        function randn() {
            var u = Math.random() || 1e-9, v = Math.random();
            return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        }
        function sigm(v) { return 1 / (1 + Math.exp(-v)); }
        function logit(x) { return w0 + w1 * x + w2 * x * x; }
        function dLogit(x) { return w1 + 2 * w2 * x; }
        function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

        function trainStep() {
            var K = +sK.value, i, k;
            for (k = 0; k < K; k++) {
                var g0 = 0, g1 = 0, g2 = 0;
                for (i = 0; i < BATCH; i++) {
                    var xr = TRUE_MU + TRUE_SIG * randn();
                    var c = (1 - sigm(logit(xr))) / BATCH;
                    g0 += c; g1 += c * xr; g2 += c * xr * xr;
                    var xf = a * randn() + b;
                    var c2 = -sigm(logit(xf)) / BATCH;
                    g0 += c2; g1 += c2 * xf; g2 += c2 * xf * xf;
                }
                w0 = clamp(w0 + LRD * g0, -15, 15);
                w1 = clamp(w1 + LRD * g1, -15, 15);
                w2 = clamp(w2 + LRD * g2, -5, 5);
            }
            var ga = 0, gb = 0;
            for (i = 0; i < BATCH; i++) {
                var z = randn(), x = a * z + b;
                var d = sigm(logit(x)), dv = dLogit(x);
                var dldx = mode === 0 ? (-d * dv) : (-(1 - d) * dv);
                gb += dldx / BATCH; ga += dldx * z / BATCH;
            }
            b = clamp(b - LRG * clamp(gb, -3, 3), -6, 7);
            a = clamp(a - LRG * clamp(ga, -3, 3), 0.05, 4);
            steps++;
        }

        function valueEstimate() {
            var V = 0, dx = 0.02;
            for (var x = -8; x <= 9; x += dx) {
                var D = clamp(sigm(logit(x)), 1e-12, 1 - 1e-12);
                V += (gaussPdf(x, TRUE_MU, TRUE_SIG) * Math.log(D) + gaussPdf(x, b, a) * Math.log(1 - D)) * dx;
            }
            return V;
        }

        function draw() {
            var w = canvas._w, h = canvas._h;
            var padL = 30, padR = 40, topY = 22, baseY = h - 46;
            var X0 = -5, X1 = 6, maxV = 0.95;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (w - padL - padR); }
            function pyD(v) { return baseY - Math.min(v / maxV, 1) * (baseY - topY); }
            function pyP(v) { return baseY - v * (baseY - topY); }

            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();

            ctx.strokeStyle = '#e2e2e2'; ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(padL, pyP(0.5)); ctx.lineTo(w - padR, pyP(0.5)); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#bbb'; ctx.font = '9.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('D = ½', w - padR + 4, pyP(0.5) + 3);

            var x, y;
            // real density, filled
            ctx.beginPath(); ctx.moveTo(px(X0), baseY);
            for (x = X0; x <= X1; x += 0.03) ctx.lineTo(px(x), pyD(gaussPdf(x, TRUE_MU, TRUE_SIG)));
            ctx.lineTo(px(X1), baseY); ctx.closePath();
            ctx.fillStyle = TEAL_FILL; ctx.fill();
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.03) {
                y = pyD(gaussPdf(x, TRUE_MU, TRUE_SIG));
                x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2; ctx.stroke();

            // fake density, filled
            ctx.beginPath(); ctx.moveTo(px(X0), baseY);
            for (x = X0; x <= X1; x += 0.03) ctx.lineTo(px(x), pyD(gaussPdf(x, b, a)));
            ctx.lineTo(px(X1), baseY); ctx.closePath();
            ctx.fillStyle = ORANGE_FILL; ctx.fill();
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.03) {
                y = pyD(gaussPdf(x, b, a));
                x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; ctx.stroke();

            // critic
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.02) {
                y = pyP(sigm(logit(x)));
                x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2.2; ctx.stroke();

            // a strip of generated samples
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
            for (var i = 0; i < 50; i++) {
                var xs = a * randn() + b;
                if (xs < X0 || xs > X1) continue;
                ctx.beginPath(); ctx.moveTo(px(xs), baseY + 6); ctx.lineTo(px(xs), baseY + 16); ctx.stroke();
            }
            ctx.globalAlpha = 1;
            ctx.fillStyle = '#999'; ctx.font = '9.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('samples the generator is producing right now', padL, baseY + 30);

            ctx.font = '11px Inter, sans-serif';
            ctx.fillStyle = TEAL; ctx.fillText('real P_x', padL + 2, 13);
            ctx.fillStyle = ORANGE; ctx.fillText('fake P_θ', padL + 66, 13);
            ctx.fillStyle = '#7c3aed'; ctx.fillText('critic D_ω', padL + 136, 13);

            r1.textContent = 'step ' + steps + '    generator:  a = ' + a.toFixed(3) +
                ' (target 0.500)   b = ' + b.toFixed(3) + ' (target 1.600)';
            r2.textContent = 'V(θ, ω) = ' + valueEstimate().toFixed(4) +
                '    perfect match would pin this at −log 4 = −1.3863';
        }

        function stop() {
            if (animId) { cancelAnimationFrame(animId); animId = null; }
            bPlay.textContent = '▶ Train';
        }

        bPlay.addEventListener('click', function () {
            if (animId) { stop(); return; }
            bPlay.textContent = '❚❚ Pause';
            function loop() {
                for (var i = 0; i < 3; i++) trainStep();
                draw();
                animId = requestAnimationFrame(loop);
            }
            animId = requestAnimationFrame(loop);
        });

        b200.addEventListener('click', function () {
            stop();
            for (var i = 0; i < 200; i++) trainStep();
            draw();
        });

        bReset.addEventListener('click', function () { stop(); reset(); draw(); });

        bMode.addEventListener('click', function () {
            mode = mode === 0 ? 1 : 0;
            bMode.textContent = mode === 0 ? 'loss: minimax' : 'loss: non-saturating';
            bMode.className = mode === 0 ? 'secondary' : '';
        });

        document.addEventListener('visibilitychange', function () {
            if (document.hidden) stop();
        });

        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 13 (Lecture 4): universal approximation, by hand
    // ─────────────────────────────────────────────────────────
    function initUat() {
        var root = byId('widget-uat');
        if (!root) return;
        makeTitle(root, 'Try it — how a ReLU network "represents" 2x³ + 3x + 5');

        var sH = slider(root, 'hidden ReLU units', 1, 24, 1, 3);
        var canvas = makeCanvas(root, 240);
        var ctx = canvas.getContext('2d');
        var readout = make('div', 'w-readout', root, '');

        make('div', 'w-note', root,
            'The network never discovers the formula 2x³ + 3x + 5. What it does is bend a straight line at a handful of ' +
            'places: each hidden ReLU unit contributes one kink, and the output layer picks how sharply to bend at each ' +
            'one. Three units give a crude zig-zag; twenty give something you cannot distinguish from the cubic by eye. ' +
            'That is the whole content of the universal approximation theorem — enough kinks approximate any continuous ' +
            'function on a bounded interval, to any accuracy you like. Two things this settles: the critic T(x) does not ' +
            'need to be convex (a cubic is not, and the fit tracks it fine — only f had to be convex, and f is a function ' +
            'we chose, not one we learn), and "parameterise the function class by a neural network" is not hand-waving. ' +
            'The weights here are solved exactly by least squares rather than by gradient descent, so what you see is the ' +
            'best that many units can do, not an artefact of an optimiser.');

        function target(x) { return 2 * x * x * x + 3 * x + 5; }

        function solve(M, rhs, k) {
            var i, j, r;
            for (i = 0; i < k; i++) {
                var piv = i;
                for (r = i + 1; r < k; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
                var tmp = M[i]; M[i] = M[piv]; M[piv] = tmp;
                var t2 = rhs[i]; rhs[i] = rhs[piv]; rhs[piv] = t2;
                var d = M[i][i];
                if (Math.abs(d) < 1e-12) continue;
                for (j = i; j < k; j++) M[i][j] /= d;
                rhs[i] /= d;
                for (r = 0; r < k; r++) {
                    if (r === i) continue;
                    var fac = M[r][i];
                    if (fac === 0) continue;
                    for (j = i; j < k; j++) M[r][j] -= fac * M[i][j];
                    rhs[r] -= fac * rhs[i];
                }
            }
            return rhs;
        }

        function fit(H) {
            var X0 = -2, X1 = 2, n = 240;
            var breaks = [];
            for (var q = 0; q < H; q++) breaks.push(X0 + (q + 1) * (X1 - X0) / (H + 1));
            var k = H + 2; // {1, x, relu(x - b_q)}
            var A = [], y = [];
            for (var i = 0; i < n; i++) {
                var x = X0 + i * (X1 - X0) / (n - 1);
                var row = [1, x];
                for (q = 0; q < H; q++) row.push(Math.max(0, x - breaks[q]));
                A.push(row); y.push(target(x));
            }
            var M = [], rhs = [];
            for (i = 0; i < k; i++) {
                var mrow = [];
                for (var j = 0; j < k; j++) {
                    var s = 0;
                    for (var r = 0; r < n; r++) s += A[r][i] * A[r][j];
                    mrow.push(s + (i === j ? 1e-7 : 0));
                }
                M.push(mrow);
                var s2 = 0;
                for (r = 0; r < n; r++) s2 += A[r][i] * y[r];
                rhs.push(s2);
            }
            var c = solve(M, rhs, k);
            return {
                breaks: breaks,
                eval: function (x) {
                    var v = c[0] + c[1] * x;
                    for (var q2 = 0; q2 < H; q2++) v += c[q2 + 2] * Math.max(0, x - breaks[q2]);
                    return v;
                }
            };
        }

        function draw() {
            var H = +sH.value;
            sH._val.textContent = H;
            var model = fit(H);

            var w = canvas._w, h = canvas._h;
            var padL = 38, padR = 16, topY = 16, baseY = h - 30;
            var X0 = -2, X1 = 2, Y0 = -14, Y1 = 26;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (w - padL - padR); }
            function py(y) { return baseY - (y - Y0) / (Y1 - Y0) * (baseY - topY); }

            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, py(0)); ctx.lineTo(w - padR, py(0)); ctx.stroke();
            ctx.strokeStyle = '#999';
            ctx.beginPath(); ctx.moveTo(px(0), topY); ctx.lineTo(px(0), baseY); ctx.stroke();

            var x, maxErr = 0, sse = 0, cnt = 0;
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.01) {
                var y = py(target(x));
                x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.6; ctx.stroke();

            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.005) {
                var fx = model.eval(x);
                var e = Math.abs(fx - target(x));
                if (e > maxErr) maxErr = e;
                sse += e * e; cnt++;
                var yy = py(fx);
                x === X0 ? ctx.moveTo(px(x), yy) : ctx.lineTo(px(x), yy);
            }
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; ctx.stroke();

            // kink locations
            ctx.strokeStyle = 'rgba(217,119,6,0.45)'; ctx.lineWidth = 1;
            model.breaks.forEach(function (bk) {
                ctx.beginPath(); ctx.moveTo(px(bk), baseY); ctx.lineTo(px(bk), baseY + 7); ctx.stroke();
            });
            ctx.fillStyle = '#999'; ctx.font = '9.5px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('one kink per hidden unit', (px(X0) + px(X1)) / 2, baseY + 22);

            ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('target  2x³ + 3x + 5', padL + 4, 13);
            ctx.fillStyle = ORANGE; ctx.fillText('ReLU network', padL + 140, 13);

            readout.textContent = H + ' hidden units    max error = ' + maxErr.toFixed(4) +
                '    RMSE = ' + Math.sqrt(sse / cnt).toFixed(4);
        }

        sH.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 14 (Lecture 4): the whole chapter on three numbers
    // ─────────────────────────────────────────────────────────
    function initToy() {
        var root = byId('widget-toy');
        if (!root) return;
        makeTitle(root, 'Try it — every theorem in this chapter, on a 3-outcome world');

        var P = [0.5, 0.3, 0.2];
        var names = ['cat', 'dog', 'blob'];

        var s1 = slider(root, 'model q(cat)', 0.05, 0.9, 0.01, 0.2);
        var s2 = slider(root, 'model q(dog)', 0.05, 0.9, 0.01, 0.3);

        var canvas = makeCanvas(root, 150);
        var ctx = canvas.getContext('2d');

        var table = make('table', 'w-table', root);
        var thead = make('thead', null, table);
        var htr = make('tr', null, thead);
        ['outcome', 'p', 'q', 'u = p/q', 'f(u)', 'q·u', 'q·f(u)', 'D* = p/(p+q)'].forEach(function (t) {
            make('th', null, htr, t);
        });
        var tbody = make('tbody', null, table);
        var rows = [0, 1, 2].map(function () {
            var tr = make('tr', null, tbody);
            return [0, 1, 2, 3, 4, 5, 6, 7].map(function () { return make('td', null, tr, ''); });
        });
        var totTr = make('tr', 'total', tbody);
        var totCells = [0, 1, 2, 3, 4, 5, 6, 7].map(function () { return make('td', null, totTr, ''); });

        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Three possible outcomes, a true distribution p fixed at (0.5, 0.3, 0.2), and a model q you control. ' +
            'Every integral in this chapter becomes a three-term sum here, so you can check the claims by hand. ' +
            'Watch the q·u column: it always totals exactly 1, no matter where you drag the sliders — that is the pinned ' +
            'average from the non-negativity proof, and it is the reason the divergence has a floor. The bottom row ' +
            'totals q·f(u), which is the definition of D_f. Then compare the three headline numbers below: the divergence ' +
            'computed straight from the definition, the value of the GAN objective at its optimal discriminator, and ' +
            '2·D_JS − log 4. They are the same number to every decimal place, which is the main theorem of §9 with the ' +
            'proof replaced by arithmetic you can verify yourself. Slide q onto (0.5, 0.3, 0.2) and everything lands on ' +
            'the perfect-match values: every ratio 1, every D* = ½, and the objective exactly −log 4.');

        function f(u) { return u <= 0 ? 0 : u * Math.log(u) - (u + 1) * Math.log(u + 1); }

        function draw() {
            var q1 = +s1.value, q2 = +s2.value;
            var q3 = 1 - q1 - q2;
            if (q3 < 0.02) { q3 = 0.02; var sc = (1 - q3) / (q1 + q2); q1 *= sc; q2 *= sc; }
            s1._val.textContent = q1.toFixed(2);
            s2._val.textContent = q2.toFixed(2);
            var Q = [q1, q2, q3];

            var sumQU = 0, Df = 0, V = 0, js = 0;
            for (var i = 0; i < 3; i++) {
                var p = P[i], q = Q[i];
                var u = p / q;
                var Dstar = p / (p + q);
                var m = 0.5 * (p + q);
                sumQU += q * u;
                Df += q * f(u);
                V += p * Math.log(Dstar) + q * Math.log(1 - Dstar);
                js += 0.5 * (p * Math.log(p / m) + q * Math.log(q / m));

                rows[i][0].textContent = names[i];
                rows[i][1].textContent = p.toFixed(3);
                rows[i][2].textContent = q.toFixed(3);
                rows[i][3].textContent = u.toFixed(3);
                rows[i][4].textContent = f(u).toFixed(4);
                rows[i][5].textContent = (q * u).toFixed(3);
                rows[i][6].textContent = (q * f(u)).toFixed(4);
                rows[i][7].textContent = Dstar.toFixed(3);
            }
            totCells[0].textContent = 'total';
            totCells[1].textContent = '1.000';
            totCells[2].textContent = '1.000';
            totCells[3].textContent = '—';
            totCells[4].textContent = '—';
            totCells[5].textContent = sumQU.toFixed(3);
            totCells[6].textContent = Df.toFixed(4);
            totCells[7].textContent = '—';

            // bars
            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var padL = 30, baseY = h - 26, topY = 14;
            var groupW = (w - padL - 20) / 3;
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL - 6, baseY); ctx.lineTo(w - 14, baseY); ctx.stroke();
            for (i = 0; i < 3; i++) {
                var gx = padL + i * groupW;
                var bw = groupW * 0.3;
                var hp = (P[i] / 0.9) * (baseY - topY);
                var hq = (Q[i] / 0.9) * (baseY - topY);
                ctx.fillStyle = TEAL_FILL; ctx.strokeStyle = TEAL; ctx.lineWidth = 1.5;
                ctx.fillRect(gx, baseY - hp, bw, hp); ctx.strokeRect(gx, baseY - hp, bw, hp);
                ctx.fillStyle = ORANGE_FILL; ctx.strokeStyle = ORANGE;
                ctx.fillRect(gx + bw + 6, baseY - hq, bw, hq); ctx.strokeRect(gx + bw + 6, baseY - hq, bw, hq);
                ctx.fillStyle = '#555'; ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'center';
                ctx.fillText(names[i], gx + bw + 3, baseY + 16);
            }
            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('true p', padL, 11);
            ctx.fillStyle = ORANGE; ctx.fillText('model q', padL + 54, 11);

            r1.textContent = 'D_f from the definition  = ' + Df.toFixed(6) +
                '        V(D*) = ' + V.toFixed(6);
            r2.textContent = '2·D_JS − log 4 = ' + (2 * js - Math.log(4)).toFixed(6) +
                '        D_JS = ' + js.toFixed(6);

            var spread = Math.max(Math.abs(Df - V), Math.abs(Df - (2 * js - Math.log(4))));
            if (Math.abs(Df + Math.log(4)) < 1e-9) {
                verdict.textContent = '✓ q = p exactly: every ratio is 1, every D* is ½, and the objective sits at −log 4 = −1.386294.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = '✓ all three agree to ' + spread.toExponential(1) +
                    ' — the definition, the optimal-discriminator value, and 2·D_JS − log 4 are the same quantity.';
                verdict.style.color = '#0e7490';
            }
        }

        [s1, s2].forEach(function (s) { s.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 15 (Lecture 5): fooling a frozen classifier
    // ─────────────────────────────────────────────────────────
    function initFixedClf() {
        var root = byId('widget-fixedclf');
        if (!root) return;
        makeTitle(root, 'Try it — fool the classifier without matching the distribution');

        var REAL = { x: -1.6, y: 1.4 };
        var gen = { x: 1.5, y: -1.3 };
        // linear classifier w0 + w1*x + w2*y ; fitted to the CURRENT gen position
        var w = [0, 0, 0];

        var sX = slider(root, 'generator centre — feature 1', -3, 3, 0.05, gen.x);
        var sY = slider(root, 'generator centre — feature 2', -3, 3, 0.05, gen.y);

        var controls = make('div', null, root);
        var bFit = make('button', null, controls, 'Retrain the classifier');
        var bAuto = make('button', 'secondary', controls, '▶ Run cat-and-mouse');
        var bReset = make('button', 'secondary', controls, 'Reset');

        var canvas = makeCanvas(root, 300);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Teal crosses are real data, fixed. Orange circles are the generator\'s output, which you move. The grey line is a ' +
            'logistic classifier, and the shading is its verdict. Start by pressing "Retrain the classifier" — it cleanly ' +
            'separates the two clusters. Now drag the generator somewhere else on the real side and watch the fooled ' +
            'percentage shoot to 100% while the distance to the real data stays enormous. That is the counter-example in ' +
            'one gesture: a frozen classifier can be defeated by moving anywhere it happens not to be looking, and defeating ' +
            'it says nothing about whether the distributions match. Press retrain again and the boundary snaps back to catch ' +
            'you. "Run cat-and-mouse" alternates the two moves automatically — the generator hops, the classifier chases, ' +
            'forever, without ever being forced onto the real data. That endless hopping is what mode collapse looks like ' +
            'from above, and it is why the classifier has to be retrained inside the loop rather than trained once.');

        var rngReal = [], rngFake = [];
        (function () {
            var i, seed = 12345;
            function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
            function randn() { return Math.sqrt(-2 * Math.log(rnd() || 1e-9)) * Math.cos(2 * Math.PI * rnd()); }
            for (i = 0; i < 26; i++) { rngReal.push([randn() * 0.42, randn() * 0.42]); rngFake.push([randn() * 0.42, randn() * 0.42]); }
        })();

        function realPts() { return rngReal.map(function (o) { return [REAL.x + o[0], REAL.y + o[1]]; }); }
        function fakePts() { return rngFake.map(function (o) { return [gen.x + o[0], gen.y + o[1]]; }); }
        function score(p) { return w[0] + w[1] * p[0] + w[2] * p[1]; }
        function sig(v) { return 1 / (1 + Math.exp(-v)); }

        function fitClassifier() {
            w = [0, 0, 0];
            var R = realPts(), F = fakePts(), n = R.length, it, i;
            for (it = 0; it < 400; it++) {
                var g = [0, 0, 0];
                for (i = 0; i < n; i++) {
                    var e = (1 - sig(score(R[i]))) / n;
                    g[0] += e; g[1] += e * R[i][0]; g[2] += e * R[i][1];
                    var e2 = -sig(score(F[i])) / n;
                    g[0] += e2; g[1] += e2 * F[i][0]; g[2] += e2 * F[i][1];
                }
                w[0] += 0.9 * g[0]; w[1] += 0.9 * g[1]; w[2] += 0.9 * g[2];
            }
        }

        function draw() {
            gen.x = +sX.value; gen.y = +sY.value;
            sX._val.textContent = gen.x.toFixed(2);
            sY._val.textContent = gen.y.toFixed(2);

            var wd = canvas._w, h = canvas._h;
            var X0 = -3.6, X1 = 3.6, Y0 = -3.0, Y1 = 3.0;
            function px(x) { return 10 + (x - X0) / (X1 - X0) * (wd - 20); }
            function py(y) { return h - 10 - (y - Y0) / (Y1 - Y0) * (h - 20); }

            ctx.clearRect(0, 0, wd, h);

            // shaded verdict regions, drawn as a coarse grid
            var step = 6;
            for (var sx = 10; sx < wd - 10; sx += step) {
                for (var sy = 10; sy < h - 10; sy += step) {
                    var dx = X0 + (sx - 10) / (wd - 20) * (X1 - X0);
                    var dy = Y0 + (h - 10 - sy) / (h - 20) * (Y1 - Y0);
                    var d = sig(w[0] + w[1] * dx + w[2] * dy);
                    ctx.fillStyle = d > 0.5
                        ? 'rgba(14,116,144,' + (0.04 + 0.10 * (d - 0.5) * 2).toFixed(3) + ')'
                        : 'rgba(217,119,6,' + (0.04 + 0.10 * (0.5 - d) * 2).toFixed(3) + ')';
                    ctx.fillRect(sx, sy, step, step);
                }
            }

            // decision boundary w0 + w1 x + w2 y = 0
            if (Math.abs(w[1]) > 1e-6 || Math.abs(w[2]) > 1e-6) {
                ctx.strokeStyle = '#555'; ctx.lineWidth = 2;
                ctx.beginPath();
                if (Math.abs(w[2]) > 1e-6) {
                    ctx.moveTo(px(X0), py(-(w[0] + w[1] * X0) / w[2]));
                    ctx.lineTo(px(X1), py(-(w[0] + w[1] * X1) / w[2]));
                } else {
                    ctx.moveTo(px(-w[0] / w[1]), py(Y0));
                    ctx.lineTo(px(-w[0] / w[1]), py(Y1));
                }
                ctx.stroke();
            }

            // real crosses
            ctx.strokeStyle = TEAL; ctx.lineWidth = 1.8;
            realPts().forEach(function (p) {
                var a = px(p[0]), b = py(p[1]);
                ctx.beginPath(); ctx.moveTo(a - 4, b - 4); ctx.lineTo(a + 4, b + 4);
                ctx.moveTo(a + 4, b - 4); ctx.lineTo(a - 4, b + 4); ctx.stroke();
            });
            // fake circles
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 1.7;
            var F = fakePts(), fooled = 0;
            F.forEach(function (p) {
                ctx.beginPath(); ctx.arc(px(p[0]), py(p[1]), 4.2, 0, 7); ctx.stroke();
                if (sig(score(p)) > 0.5) fooled++;
            });

            ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('real  P_x', 14, 16);
            ctx.fillStyle = ORANGE; ctx.fillText('fake  P_θ', 86, 16);
            ctx.fillStyle = '#666'; ctx.fillText('shading = classifier verdict', 168, 16);

            var pct = Math.round(100 * fooled / F.length);
            var dist = Math.sqrt(Math.pow(gen.x - REAL.x, 2) + Math.pow(gen.y - REAL.y, 2));
            r1.textContent = 'fakes accepted as real:  ' + pct + '%';
            r2.textContent = 'distance between cluster centres:  ' + dist.toFixed(2) +
                '   (0.00 would mean P_θ = P_x)';

            if (pct >= 85 && dist > 1.2) {
                verdict.textContent = '✗ Classifier fooled ' + pct + '% of the time — and the distributions are still ' +
                    dist.toFixed(2) + ' apart. Fooling it proved nothing.';
                verdict.style.color = '#c0392b';
            } else if (dist < 0.45) {
                verdict.textContent = '✓ Now the clusters genuinely overlap. This is the only way to fool a classifier that keeps retraining.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = 'The classifier is catching ' + (100 - pct) + '% of the fakes. Move the generator to escape it.';
                verdict.style.color = '#999';
            }
        }

        function escapeStep() {
            // move the generator along the direction that most increases the classifier's score
            var n = Math.sqrt(w[1] * w[1] + w[2] * w[2]) || 1;
            sX.value = Math.max(-3, Math.min(3, gen.x + 1.5 * w[1] / n));
            sY.value = Math.max(-3, Math.min(3, gen.y + 1.5 * w[2] / n));
            draw();
        }

        var autoId = null, phase = 0;
        function stopAuto() {
            if (autoId) { clearInterval(autoId); autoId = null; }
            bAuto.textContent = '▶ Run cat-and-mouse';
        }

        bFit.addEventListener('click', function () { fitClassifier(); draw(); });
        bAuto.addEventListener('click', function () {
            if (autoId) { stopAuto(); return; }
            bAuto.textContent = '❚❚ Stop';
            autoId = setInterval(function () {
                if (phase === 0) { fitClassifier(); } else { escapeStep(); }
                phase = 1 - phase;
                draw();
            }, 700);
        });
        bReset.addEventListener('click', function () {
            stopAuto(); sX.value = 1.5; sY.value = -1.3; w = [0, 0, 0]; draw();
        });
        [sX, sY].forEach(function (s) { s.addEventListener('input', function () { stopAuto(); draw(); }); });

        fitClassifier();
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 16 (Lecture 5): mode collapse and discriminator lag
    // ─────────────────────────────────────────────────────────
    function initModeCollapse() {
        var root = byId('widget-modecollapse');
        if (!root) return;
        makeTitle(root, 'Try it — mode collapse, and the lag that causes the hopping');

        var MODES = [[-1.4, 1.4], [1.4, 1.4], [-1.4, -1.4], [1.4, -1.4]];
        var SIG = 0.42;
        var blobs = [], steps = 0, animId = null;

        var sK = slider(root, 'generator capacity (number of blobs)', 1, 4, 1, 1);

        var controls = make('div', null, root);
        var bPlay = make('button', null, controls, '▶ Train');
        var bReset = make('button', 'secondary', controls, 'Reset');

        var canvas = makeCanvas(root, 300);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'The real data has four modes (teal rings). The generator is a mixture of k blobs (orange) whose centres move ' +
            'by gradient ascent on log D — the non-saturating generator loss from Ch.4. The discriminator here is not a ' +
            'network at all: it is the exact optimal critic D* = p/(p+q) computed analytically, which is the strongest ' +
            'possible critic and removes any question of the bound being loose. Set capacity to 1 and press train: the ' +
            'single blob climbs onto one mode and stops there, permanently, with three quarters of the data distribution ' +
            'simply abandoned. That is mode collapse in its purest form, and notice it is not a bug or a bad learning ' +
            'rate — the generator has reached a genuine stationary point of a perfectly optimised objective. It cannot ' +
            'cover four modes because it does not have the capacity to be four things at once. Raise k and the blobs ' +
            'repel each other (two blobs on one mode double q there, which drops D and pushes one away) until, at k = 4, ' +
            'every mode is claimed and the shading goes flat at ½ — the signature of a generator that has actually won.');

        function reset() {
            // start the blobs near the centre but at distinct offsets: identical starts would leave
            // them symmetric forever and they would all climb to the same mode
            var seed = [[0.10, 0.05], [-0.12, 0.08], [0.06, -0.11], [-0.05, -0.07]];
            blobs = seed.map(function (s) {
                return [s[0] + (Math.random() - 0.5) * 0.06, s[1] + (Math.random() - 0.5) * 0.06];
            });
            steps = 0;
        }
        reset();

        function g2(x, y, mx, my, s) {
            return Math.exp(-((x - mx) * (x - mx) + (y - my) * (y - my)) / (2 * s * s)) / (2 * Math.PI * s * s);
        }
        function pReal(x, y) {
            var t = 0;
            for (var i = 0; i < 4; i++) t += g2(x, y, MODES[i][0], MODES[i][1], SIG);
            return t / 4;
        }
        function pFake(x, y, cs) {
            var k = cs.length, t = 0;
            for (var i = 0; i < k; i++) t += g2(x, y, cs[i][0], cs[i][1], SIG);
            return t / k;
        }
        function activeBlobs() { return blobs.slice(0, +sK.value); }
        function D(x, y) {
            var p = pReal(x, y), q = pFake(x, y, activeBlobs());
            return p / (p + q + 1e-300);
        }

        function trainStep() {
            var cs = activeBlobs().map(function (b) { return [b[0], b[1]]; });
            var eps = 0.02, lr = 0.06;
            for (var i = 0; i < cs.length; i++) {
                var x = cs[i][0], y = cs[i][1];
                // ascend log D, estimated by central differences
                var gx = (Math.log(D(x + eps, y)) - Math.log(D(x - eps, y))) / (2 * eps);
                var gy = (Math.log(D(x, y + eps)) - Math.log(D(x, y - eps))) / (2 * eps);
                var n = Math.sqrt(gx * gx + gy * gy);
                if (n > 6) { gx = gx * 6 / n; gy = gy * 6 / n; }
                blobs[i][0] = Math.max(-3, Math.min(3, x + lr * gx));
                blobs[i][1] = Math.max(-3, Math.min(3, y + lr * gy));
            }
            steps++;
        }

        function coverage() {
            var cs = activeBlobs(), hit = 0;
            for (var m = 0; m < 4; m++) {
                for (var i = 0; i < cs.length; i++) {
                    var d = Math.hypot(cs[i][0] - MODES[m][0], cs[i][1] - MODES[m][1]);
                    if (d < 0.7) { hit++; break; }
                }
            }
            return hit;
        }

        function draw() {
            var wd = canvas._w, h = canvas._h;
            var X0 = -3, X1 = 3, Y0 = -2.6, Y1 = 2.6;
            function px(x) { return 10 + (x - X0) / (X1 - X0) * (wd - 20); }
            function py(y) { return h - 10 - (y - Y0) / (Y1 - Y0) * (h - 20); }
            ctx.clearRect(0, 0, wd, h);

            // critic field
            var step = 7;
            for (var sx = 10; sx < wd - 10; sx += step) {
                for (var sy = 10; sy < h - 10; sy += step) {
                    var dx = X0 + (sx - 10) / (wd - 20) * (X1 - X0);
                    var dy = Y0 + (h - 10 - sy) / (h - 20) * (Y1 - Y0);
                    var d = D(dx, dy);
                    ctx.fillStyle = d > 0.5
                        ? 'rgba(14,116,144,' + (0.05 + 0.16 * (d - 0.5) * 2).toFixed(3) + ')'
                        : 'rgba(217,119,6,' + (0.05 + 0.16 * (0.5 - d) * 2).toFixed(3) + ')';
                    ctx.fillRect(sx, sy, step, step);
                }
            }

            // real modes
            ctx.strokeStyle = TEAL; ctx.lineWidth = 1.8;
            MODES.forEach(function (m) {
                ctx.beginPath(); ctx.arc(px(m[0]), py(m[1]), 22, 0, 7); ctx.stroke();
                ctx.beginPath(); ctx.arc(px(m[0]), py(m[1]), 3, 0, 7);
                ctx.fillStyle = TEAL; ctx.fill();
            });

            // generator blobs
            activeBlobs().forEach(function (b) {
                ctx.fillStyle = 'rgba(217,119,6,0.30)';
                ctx.beginPath(); ctx.arc(px(b[0]), py(b[1]), 20, 0, 7); ctx.fill();
                ctx.strokeStyle = ORANGE; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.arc(px(b[0]), py(b[1]), 20, 0, 7); ctx.stroke();
            });

            ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('4 real modes', 14, 16);
            ctx.fillStyle = ORANGE; ctx.fillText('generator blobs', 104, 16);

            var cov = coverage();
            r1.textContent = 'step ' + steps + '    capacity k = ' + sK.value +
                '    modes covered: ' + cov + ' / 4';
            if (cov === 4) {
                verdict.textContent = '✓ All four modes covered — the generator has actually matched the distribution.';
                verdict.style.color = '#0e7490';
            } else if (+sK.value === 1) {
                verdict.textContent = '✗ Mode collapse: one blob cannot be four modes, so ' + (4 - cov) + ' of them are simply abandoned.';
                verdict.style.color = '#c0392b';
            } else {
                verdict.textContent = 'Covering ' + cov + ' of 4. Blobs sitting on the same mode leave the others empty.';
                verdict.style.color = '#999';
            }
        }

        function stop() { if (animId) { cancelAnimationFrame(animId); animId = null; } bPlay.textContent = '▶ Train'; }

        bPlay.addEventListener('click', function () {
            if (animId) { stop(); return; }
            bPlay.textContent = '❚❚ Pause';
            function loop() { trainStep(); draw(); animId = requestAnimationFrame(loop); }
            animId = requestAnimationFrame(loop);
        });
        bReset.addEventListener('click', function () { stop(); reset(); draw(); });
        sK.addEventListener('input', function () { sK._val.textContent = sK.value; draw(); });
        document.addEventListener('visibilitychange', function () { if (document.hidden) stop(); });

        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 17 (Lecture 5): y as a control switch
    // ─────────────────────────────────────────────────────────
    function initCgan() {
        var root = byId('widget-cgan');
        if (!root) return;
        makeTitle(root, 'Try it — what the conditioning variable actually buys you');

        var CLASSES = [
            { name: 'class 0', c: '#0e7490', mx: -1.5, my: 1.1 },
            { name: 'class 1', c: '#d97706', mx: 1.5, my: 1.1 },
            { name: 'class 2', c: '#7c3aed', mx: -1.5, my: -1.1 },
            { name: 'class 3', c: '#c0392b', mx: 1.5, my: -1.1 }
        ];
        var mode = 'cond', chosen = 0, samples = [];

        var controls = make('div', null, root);
        var bUncond = make('button', 'secondary', controls, 'Unconditional G(z)');
        var bCond = make('button', null, controls, 'Conditional G(z, y)');

        var chipbox = make('div', 'w-chipbox', root);
        var chips = CLASSES.map(function (cl, i) {
            var c = make('span', i === 0 ? 'chip on' : 'chip', chipbox, 'y = ' + i);
            c.addEventListener('click', function () {
                chosen = i;
                chips.forEach(function (cc, j) { cc.className = j === i ? 'chip on' : 'chip'; });
                draw();
            });
            return c;
        });

        var bDraw = make('button', null, make('div', null, root), 'Draw 12 samples');
        var canvas = makeCanvas(root, 250);
        var ctx = canvas.getContext('2d');
        var readout = make('div', 'w-readout', root, '');

        make('div', 'w-note', root,
            'A stand-in for a class-conditional generator: four classes, each occupying its own region of the output space. ' +
            'In unconditional mode the only input is noise, so samples land wherever the marginal distribution says — ' +
            'realistic, but you get whatever you get, and there is no input you could have set to ask for class 2. ' +
            'Switch to conditional and the label becomes a switch: pick y, draw fresh noise, and every sample lands in that ' +
            'class. The division of labour is the thing to remember — z decides which sample you get, y decides which kind. ' +
            'Nothing about the adversarial game changes; the label is simply concatenated onto the inputs of both networks, ' +
            'and the discriminator now judges the pair (x, y) rather than x alone, which is what stops the generator from ' +
            'ignoring the label and emitting any realistic image it likes.');

        function randn() { var u = Math.random() || 1e-9, v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

        function sample() {
            samples = [];
            for (var i = 0; i < 12; i++) {
                var k = mode === 'cond' ? chosen : Math.floor(Math.random() * 4);
                samples.push([CLASSES[k].mx + randn() * 0.34, CLASSES[k].my + randn() * 0.28, k]);
            }
        }
        sample();

        function draw() {
            var wd = canvas._w, h = canvas._h;
            var X0 = -3, X1 = 3, Y0 = -2.2, Y1 = 2.2;
            function px(x) { return 12 + (x - X0) / (X1 - X0) * (wd - 24); }
            function py(y) { return h - 12 - (y - Y0) / (Y1 - Y0) * (h - 24); }
            ctx.clearRect(0, 0, wd, h);

            CLASSES.forEach(function (cl, i) {
                var on = mode === 'cond' && i === chosen;
                ctx.strokeStyle = on ? cl.c : '#ddd';
                ctx.setLineDash([5, 4]); ctx.lineWidth = on ? 1.8 : 1.2;
                ctx.beginPath(); ctx.ellipse(px(cl.mx), py(cl.my), 62, 40, 0, 0, 7); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = on ? cl.c : '#bbb';
                ctx.font = (on ? '600 ' : '') + '10.5px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(cl.name, px(cl.mx), py(cl.my) - 46);
            });

            samples.forEach(function (s) {
                ctx.fillStyle = CLASSES[s[2]].c;
                ctx.beginPath(); ctx.arc(px(s[0]), py(s[1]), 4.6, 0, 7); ctx.fill();
            });

            if (mode === 'cond') {
                readout.textContent = 'y = ' + chosen + ' held fixed, z redrawn each time  →  12 / 12 samples in the requested class';
            } else {
                var counts = [0, 0, 0, 0];
                samples.forEach(function (s) { counts[s[2]]++; });
                readout.textContent = 'no control input  →  samples scattered across classes: ' + counts.join(', ');
            }
        }

        bUncond.addEventListener('click', function () {
            mode = 'uncond'; bUncond.className = ''; bCond.className = 'secondary';
            chipbox.style.opacity = '0.35';
            sample(); draw();
        });
        bCond.addEventListener('click', function () {
            mode = 'cond'; bCond.className = ''; bUncond.className = 'secondary';
            chipbox.style.opacity = '1';
            sample(); draw();
        });
        bDraw.addEventListener('click', function () { sample(); draw(); });

        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 18 (Lecture 5): cycle consistency
    // ─────────────────────────────────────────────────────────
    function initCycle() {
        var root = byId('widget-cycle');
        if (!root) return;
        makeTitle(root, 'Try it — what the cycle loss is actually holding in place');

        var PATTERNS = [
            [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]],
            [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
        ];
        var COLLAPSED = [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]];
        var pick = 0, useCycle = true;

        var chipbox = make('div', 'w-chipbox', root);
        var chips = PATTERNS.map(function (_, i) {
            var c = make('span', i === 0 ? 'chip on' : 'chip', chipbox, 'source ' + (i + 1));
            c.addEventListener('click', function () {
                pick = i;
                chips.forEach(function (cc, j) { cc.className = j === i ? 'chip on' : 'chip'; });
                draw();
            });
            return c;
        });
        var bCyc = make('button', null, make('div', null, root), 'cycle loss: ON');

        var canvas = makeCanvas(root, 220);
        var ctx = canvas.getContext('2d');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'A deliberately tiny stand-in for image translation: a 4×4 "image", where the pattern is the content and the ' +
            'colour is the domain. G₁ has to repaint teal into orange. Turn the cycle loss off and watch what the ' +
            'adversarial loss alone permits — every source, whatever its pattern, gets mapped to the same perfectly ' +
            'convincing orange image. D_T is entirely satisfied, because that output really does look like a member of the ' +
            'target domain; nothing in the adversarial objective ever asked the output to have anything to do with the ' +
            'input. Turn the cycle loss on and the round trip has to reproduce the original, which is impossible if G₁ ' +
            'threw the pattern away. Structure is preserved not because we asked for it directly, but because destroying ' +
            'it would make the return journey unaffordable. (Scripted illustration of the failure mode, not a trained ' +
            'model — the two networks here are hand-written maps.)');

        function G1(grid) { return useCycle ? grid : COLLAPSED; }
        function G2(grid) { return useCycle ? grid : COLLAPSED; }

        function drawGrid(x0, y0, cell, grid, colour, label, sub, dashed) {
            var i, j;
            for (i = 0; i < 4; i++) {
                for (j = 0; j < 4; j++) {
                    ctx.fillStyle = grid[i][j] ? colour : '#f4f4f4';
                    ctx.fillRect(x0 + j * cell, y0 + i * cell, cell - 1.5, cell - 1.5);
                }
            }
            ctx.strokeStyle = dashed ? '#bbb' : '#ddd';
            ctx.lineWidth = 1;
            if (dashed) ctx.setLineDash([4, 3]);
            ctx.strokeRect(x0 - 2, y0 - 2, cell * 4 + 2, cell * 4 + 2);
            ctx.setLineDash([]);
            ctx.font = '600 10.5px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillStyle = '#444';
            ctx.fillText(label, x0 + cell * 2, y0 - 10);
            ctx.font = '9.4px Inter, sans-serif'; ctx.fillStyle = '#999';
            ctx.fillText(sub, x0 + cell * 2, y0 + cell * 4 + 16);
        }

        function draw() {
            var wd = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, wd, h);
            var cell = 20, y0 = 44;
            var src = PATTERNS[pick];
            var mid = G1(src);
            var back = G2(mid);
            var xs = [26, 210, 394];
            var step = Math.min(184, (wd - 120) / 3 + 40);
            xs = [26, 26 + step, 26 + 2 * step];

            drawGrid(xs[0], y0, cell, src, TEAL, 'source  s', 'domain S', false);
            drawGrid(xs[1], y0, cell, mid, ORANGE, 'G₁(s)', 'domain T', false);
            drawGrid(xs[2], y0, cell, back, TEAL, 'G₂(G₁(s))', 'back in S', true);

            ctx.strokeStyle = '#888'; ctx.lineWidth = 1.5;
            [0, 1].forEach(function (i) {
                var ax = xs[i] + cell * 4 + 8, bx = xs[i + 1] - 10, my = y0 + cell * 2;
                ctx.beginPath(); ctx.moveTo(ax, my); ctx.lineTo(bx, my); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(bx, my); ctx.lineTo(bx - 6, my - 4); ctx.lineTo(bx - 6, my + 4); ctx.closePath();
                ctx.fillStyle = '#888'; ctx.fill();
                ctx.font = '9.6px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = '#888';
                ctx.fillText(i === 0 ? 'G₁' : 'G₂', (ax + bx) / 2, my - 8);
            });

            // does the round trip match?
            var same = JSON.stringify(src) === JSON.stringify(back);
            ctx.font = '600 11px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('D_T verdict: looks like domain T ✓', 26, 22);
            ctx.fillStyle = same ? TEAL : '#c0392b';
            ctx.textAlign = 'right';
            ctx.fillText(same ? 'round trip: ‖G₂(G₁(s)) − s‖₁ = 0 ✓' : 'round trip: ‖G₂(G₁(s)) − s‖₁ is large ✗', wd - 20, 22);

            if (useCycle) {
                verdict.textContent = '✓ Cycle loss on: the pattern survives the trip, so G₁ is forced to translate rather than replace.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = '✗ Cycle loss off: all three sources collapse to one output. The discriminator is happy, the translation is worthless.';
                verdict.style.color = '#c0392b';
            }
        }

        bCyc.addEventListener('click', function () {
            useCycle = !useCycle;
            bCyc.textContent = useCycle ? 'cycle loss: ON' : 'cycle loss: OFF';
            bCyc.className = useCycle ? '' : 'secondary';
            draw();
        });

        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 19 (Lecture 5): transpose convolution arithmetic
    // ─────────────────────────────────────────────────────────
    function initTransConv() {
        var root = byId('widget-transconv');
        if (!root) return;
        makeTitle(root, 'Try it — how a transpose convolution grows an image');

        var sIn = slider(root, 'input size', 2, 6, 1, 3);
        var sK = slider(root, 'kernel size', 2, 5, 1, 3);
        var sS = slider(root, 'stride', 1, 3, 1, 2);
        var sP = slider(root, 'padding', 0, 2, 1, 1);

        var canvas = makeCanvas(root, 250);
        var ctx = canvas.getContext('2d');
        var readout = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'A normal convolution shrinks a grid; a transpose convolution is the same wiring run backwards, so it grows one. ' +
            'Each input cell is multiplied by the whole kernel and that little patch is *added into* the output, with stride ' +
            'controlling how far apart the patches are stamped. Output size follows a formula worth memorising, since ' +
            'getting it wrong is the most common reason a DCGAN refuses to compile: out = (in − 1)·stride − 2·padding + ' +
            'kernel. Drag stride to see the output grow roughly by that factor — this is exactly how z at 1×1 becomes an ' +
            'image at 100×100 over a handful of layers. One thing to notice: when stride does not divide evenly into the ' +
            'kernel, some output cells receive contributions from more inputs than their neighbours do. That uneven ' +
            'overlap is where checkerboard artefacts in GAN samples come from.');

        function draw() {
            var iN = +sIn.value, k = +sK.value, st = +sS.value, pd = +sP.value;
            // overlap counts on the (unpadded) full canvas, then crop by padding
            var full = (iN - 1) * st + k;
            // padding can't eat the whole output — frameworks require it to leave at least one cell
            var pdMax = Math.floor((full - 1) / 2);
            var clamped = pd > pdMax;
            if (clamped) pd = pdMax;
            sIn._val.textContent = iN; sK._val.textContent = k;
            sS._val.textContent = st; sP._val.textContent = pd + (clamped ? ' (capped)' : '');
            var out = full - 2 * pd;

            var wd = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, wd, h);

            var counts = [];
            var i, j, a, b;
            for (i = 0; i < full; i++) { counts.push(new Array(full)); for (j = 0; j < full; j++) counts[i][j] = 0; }
            for (i = 0; i < iN; i++) {
                for (j = 0; j < iN; j++) {
                    for (a = 0; a < k; a++) for (b = 0; b < k; b++) counts[i * st + a][j * st + b]++;
                }
            }

            var inCell = Math.min(26, 110 / iN);
            var outCell = Math.min(20, 150 / Math.max(out, 1));
            var y0 = 54;

            // input grid
            var ix = 34;
            ctx.font = '600 10.5px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = '#0e7490';
            ctx.fillText('input  ' + iN + ' × ' + iN, ix + inCell * iN / 2, y0 - 14);
            for (i = 0; i < iN; i++) for (j = 0; j < iN; j++) {
                ctx.fillStyle = 'rgba(14,116,144,0.16)';
                ctx.fillRect(ix + j * inCell, y0 + i * inCell, inCell - 2, inCell - 2);
                ctx.strokeStyle = TEAL; ctx.lineWidth = 1;
                ctx.strokeRect(ix + j * inCell, y0 + i * inCell, inCell - 2, inCell - 2);
            }

            // arrow + kernel
            var kx = ix + inCell * iN + 30;
            ctx.strokeStyle = '#888'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(kx - 22, y0 + 30); ctx.lineTo(kx - 6, y0 + 30); ctx.stroke();
            ctx.font = '600 10.5px Inter, sans-serif'; ctx.fillStyle = '#7c3aed'; ctx.textAlign = 'center';
            ctx.fillText('kernel ' + k + '×' + k, kx + k * 14 / 2, y0 - 14);
            for (i = 0; i < k; i++) for (j = 0; j < k; j++) {
                ctx.fillStyle = 'rgba(124,58,237,0.18)';
                ctx.fillRect(kx + j * 14, y0 + i * 14, 12, 12);
                ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 0.9;
                ctx.strokeRect(kx + j * 14, y0 + i * 14, 12, 12);
            }

            // output grid, coloured by overlap count
            var ox = kx + k * 14 + 44;
            ctx.font = '600 10.5px Inter, sans-serif'; ctx.fillStyle = '#d97706'; ctx.textAlign = 'center';
            ctx.fillText('output  ' + out + ' × ' + out, ox + outCell * out / 2, y0 - 14);
            var maxC = 1;
            for (i = 0; i < full; i++) for (j = 0; j < full; j++) maxC = Math.max(maxC, counts[i][j]);
            for (i = 0; i < out; i++) {
                for (j = 0; j < out; j++) {
                    var c = counts[i + pd] && counts[i + pd][j + pd] ? counts[i + pd][j + pd] : 0;
                    ctx.fillStyle = 'rgba(217,119,6,' + (0.10 + 0.55 * (c / maxC)).toFixed(3) + ')';
                    ctx.fillRect(ox + j * outCell, y0 + i * outCell, outCell - 2, outCell - 2);
                    ctx.strokeStyle = 'rgba(217,119,6,0.55)'; ctx.lineWidth = 0.8;
                    ctx.strokeRect(ox + j * outCell, y0 + i * outCell, outCell - 2, outCell - 2);
                }
            }

            ctx.font = '9.6px Inter, sans-serif'; ctx.textAlign = 'left'; ctx.fillStyle = '#999';
            ctx.fillText('darker = more input cells contributed here', ox, y0 + outCell * out + 20);

            ctx.font = '11px Inter, sans-serif'; ctx.fillStyle = '#444'; ctx.textAlign = 'center';
            ctx.fillText('out = (in − 1)·stride − 2·padding + kernel', wd / 2, h - 12);

            readout.textContent = 'out = (' + iN + ' − 1)·' + st + ' − 2·' + pd + ' + ' + k + ' = ' + out +
                '     scale-up: ' + (out / iN).toFixed(2) + '×';

            var uneven = (k % st !== 0);
            verdict.textContent = uneven
                ? '⚠ kernel ' + k + ' is not divisible by stride ' + st + ' — uneven overlap, the classic source of checkerboard artefacts.'
                : '✓ kernel divides evenly by stride — every output cell gets the same number of contributions.';
            verdict.style.color = uneven ? '#d97706' : '#0e7490';
        }

        [sIn, sK, sS, sP].forEach(function (s) { s.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 20 (Lecture 6): JS saturates, Wasserstein does not
    // The exact WGAN Example 1: two parallel unit segments offset by theta.
    // ─────────────────────────────────────────────────────────
    function initJsVsW() {
        var root = byId('widget-jsvsw');
        if (!root) return;
        makeTitle(root, 'Try it — slide the supports apart and watch JS give up');

        var sT = slider(root, 'separation θ', -1, 1, 0.01, 0.6);
        var canvas = makeCanvas(root, 250);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'This is not a simulation — it is the exact example from the WGAN paper, where every quantity has a closed ' +
            'form. P₀ is uniform on the vertical segment at x = 0; P_θ is the same segment shifted to x = θ. Left: the two ' +
            'supports. Right: three ways of measuring how far apart they are, plotted against θ, with your current value ' +
            'marked. The moment θ leaves zero the supports become disjoint, and every f-divergence immediately jumps to ' +
            'its ceiling and stays there: JS pins at log 2, total variation at 1, KL at infinity. They are flat, so their ' +
            'gradient with respect to θ is exactly zero — a generator told "you are log 2 away" learns nothing about ' +
            'which direction to move. The Wasserstein distance is simply |θ|: it knows the answer at every separation, ' +
            'and its slope points home. That single picture is the whole argument for replacing the divergence.');

        function draw() {
            var t = +sT.value;
            sT._val.textContent = t.toFixed(2);
            var W = Math.abs(t);
            var JS = Math.abs(t) < 1e-9 ? 0 : Math.log(2);

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var colW = (w - 46) / 2;

            // ── left: the two supports in the plane ──
            var padL = 30, baseY = h - 40, topY = 28;
            function pxL(x) { return padL + (x + 1.3) / 2.6 * (colW - padL - 10); }
            ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(colW - 10, baseY); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(pxL(0), topY - 6); ctx.lineTo(pxL(0), baseY + 6); ctx.stroke();

            ctx.strokeStyle = TEAL; ctx.lineWidth = 5;
            ctx.beginPath(); ctx.moveTo(pxL(0), topY); ctx.lineTo(pxL(0), baseY); ctx.stroke();
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 5;
            ctx.beginPath(); ctx.moveTo(pxL(t), topY); ctx.lineTo(pxL(t), baseY); ctx.stroke();

            if (Math.abs(t) > 0.02) {
                ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]);
                var my = (topY + baseY) / 2;
                ctx.beginPath(); ctx.moveTo(pxL(0), my); ctx.lineTo(pxL(t), my); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = '#7c3aed'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
                ctx.fillText('|θ|', (pxL(0) + pxL(t)) / 2, my - 6);
            }
            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('P₀', pxL(0) - 18, topY - 10);
            ctx.fillStyle = ORANGE; ctx.fillText('P_θ', pxL(t) + 6, topY - 10);
            ctx.fillStyle = '#999'; ctx.textAlign = 'center';
            ctx.fillText('the two supports', colW / 2, h - 16);

            // ── right: the metrics against theta ──
            var x0 = colW + 46, x1 = w - 14;
            var yTop = 30, yBot = h - 40, Vmax = 1.15;
            function pxR(tt) { return x0 + (tt + 1) / 2 * (x1 - x0); }
            function pyR(v) { return yBot - Math.min(v / Vmax, 1) * (yBot - yTop); }
            ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x0, yBot); ctx.lineTo(x1, yBot); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(pxR(0), yTop - 4); ctx.lineTo(pxR(0), yBot + 4); ctx.stroke();

            // TV = 1 for theta != 0
            ctx.strokeStyle = 'rgba(200,60,60,0.45)'; ctx.lineWidth = 1.8;
            ctx.beginPath(); ctx.moveTo(x0, pyR(1)); ctx.lineTo(pxR(-0.005), pyR(1)); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(pxR(0.005), pyR(1)); ctx.lineTo(x1, pyR(1)); ctx.stroke();
            // JS = log 2
            ctx.strokeStyle = '#c0392b'; ctx.lineWidth = 2.4;
            ctx.beginPath(); ctx.moveTo(x0, pyR(Math.log(2))); ctx.lineTo(pxR(-0.005), pyR(Math.log(2))); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(pxR(0.005), pyR(Math.log(2))); ctx.lineTo(x1, pyR(Math.log(2))); ctx.stroke();
            ctx.fillStyle = '#111';
            ctx.beginPath(); ctx.arc(pxR(0), pyR(0), 3.6, 0, 7); ctx.fill();
            // W = |theta|
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.6;
            ctx.beginPath(); ctx.moveTo(pxR(-1), pyR(1)); ctx.lineTo(pxR(0), pyR(0)); ctx.lineTo(pxR(1), pyR(1)); ctx.stroke();

            // current position
            ctx.strokeStyle = 'rgba(124,58,237,0.55)'; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
            ctx.beginPath(); ctx.moveTo(pxR(t), yTop - 4); ctx.lineTo(pxR(t), yBot + 4); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = TEAL;
            ctx.beginPath(); ctx.arc(pxR(t), pyR(W), 4.4, 0, 7); ctx.fill();
            ctx.fillStyle = '#c0392b';
            ctx.beginPath(); ctx.arc(pxR(t), pyR(JS), 4.4, 0, 7); ctx.fill();

            ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('W = |θ|', x0 + 6, yTop + 2);
            ctx.fillStyle = '#c0392b'; ctx.fillText('JS = log 2', x0 + 76, pyR(Math.log(2)) - 6);
            ctx.fillStyle = 'rgba(200,60,60,0.7)'; ctx.fillText('TV = 1', x0 + 76, pyR(1) - 6);
            ctx.fillStyle = '#999'; ctx.textAlign = 'center';
            ctx.fillText('θ', (x0 + x1) / 2, h - 16);

            r1.textContent = 'W(P₀, P_θ) = ' + W.toFixed(4) +
                '        dW/dθ = ' + (Math.abs(t) < 1e-9 ? 'undefined at 0' : (t > 0 ? '+1' : '−1'));
            r2.textContent = 'D_JS = ' + JS.toFixed(4) + '        dJS/dθ = ' +
                (Math.abs(t) < 1e-9 ? '—' : '0') + '        D_KL = ' + (Math.abs(t) < 1e-9 ? '0' : '+∞');

            if (Math.abs(t) < 1e-9) {
                verdict.textContent = '✓ θ = 0: supports coincide, every metric is 0. This is the only point where the f-divergences carry information.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = '✗ Supports disjoint. JS is stuck at 0.6931 with zero slope — no gradient — while W = ' +
                    W.toFixed(2) + ' still points the way home.';
                verdict.style.color = '#c0392b';
            }
        }

        sT.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 21 (Lecture 6): optimal transport in 1-D via the CDF gap
    // ─────────────────────────────────────────────────────────
    function initW1() {
        var root = byId('widget-w1');
        if (!root) return;
        makeTitle(root, 'Try it — the cheapest way to move the dirt');

        var BINS = 9;
        var P = [0.02, 0.06, 0.16, 0.28, 0.24, 0.14, 0.06, 0.03, 0.01];
        var sMu = slider(root, 'centre of Q', 0, 8, 0.05, 6.2);
        var sSig = slider(root, 'spread of Q', 0.5, 3.5, 0.05, 1.1);

        var canvas = makeCanvas(root, 270);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Two piles of dirt on nine positions. In one dimension the optimal transport plan has a closed form — you ' +
            'never need to search over plans at all — and the minimum work turns out to equal the area between the two ' +
            'cumulative curves, which is what the shaded band in the lower panel is. Drag Q around and watch that area, ' +
            'and the number, respond continuously: slide Q one bin further away and W₁ grows by exactly the mass you had ' +
            'to carry that extra distance. Compare that with total variation, printed alongside: once the two piles stop ' +
            'overlapping at all, TV is pinned at 1 and stops responding entirely, while W₁ keeps counting. Overlap-based ' +
            'measures answer "do these touch?"; transport-based ones answer "how far apart are they?", and only the ' +
            'second question has a useful derivative.');

        function qWeights() {
            var mu = +sMu.value, sig = +sSig.value, q = [], i, s = 0;
            for (i = 0; i < BINS; i++) { var v = Math.exp(-0.5 * Math.pow((i - mu) / sig, 2)); q.push(v); s += v; }
            for (i = 0; i < BINS; i++) q[i] /= s;
            return q;
        }

        function draw() {
            var mu = +sMu.value, sig = +sSig.value;
            sMu._val.textContent = mu.toFixed(2);
            sSig._val.textContent = sig.toFixed(2);
            var Q = qWeights();

            var cp = 0, cq = 0, W1 = 0, TV = 0, i;
            var cumP = [], cumQ = [];
            for (i = 0; i < BINS; i++) {
                cp += P[i]; cq += Q[i];
                cumP.push(cp); cumQ.push(cq);
                if (i < BINS - 1) W1 += Math.abs(cp - cq);   // bin width 1
                TV += 0.5 * Math.abs(P[i] - Q[i]);
            }

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var padL = 34, padR = 16;
            var bw = (w - padL - padR) / BINS;
            function bx(i) { return padL + i * bw; }

            // ── top: the two histograms ──
            var hTop = 40, hBot = 132, maxP = 0.35;
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, hBot); ctx.lineTo(w - padR, hBot); ctx.stroke();
            for (i = 0; i < BINS; i++) {
                var ph = Math.min(P[i] / maxP, 1) * (hBot - hTop);
                var qh = Math.min(Q[i] / maxP, 1) * (hBot - hTop);
                ctx.fillStyle = TEAL_FILL; ctx.strokeStyle = TEAL; ctx.lineWidth = 1.4;
                ctx.fillRect(bx(i) + 2, hBot - ph, bw * 0.42, ph);
                ctx.strokeRect(bx(i) + 2, hBot - ph, bw * 0.42, ph);
                ctx.fillStyle = ORANGE_FILL; ctx.strokeStyle = ORANGE;
                ctx.fillRect(bx(i) + bw * 0.5, hBot - qh, bw * 0.42, qh);
                ctx.strokeRect(bx(i) + bw * 0.5, hBot - qh, bw * 0.42, qh);
                ctx.fillStyle = '#aaa'; ctx.font = '9px Inter, sans-serif'; ctx.textAlign = 'center';
                ctx.fillText(i, bx(i) + bw / 2, hBot + 12);
            }
            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('P — the pile you have', padL, 14);
            ctx.fillStyle = ORANGE; ctx.fillText('Q — the shape you want', padL + 150, 14);

            // ── bottom: CDFs with the gap shaded ──
            var cTop = 168, cBot = h - 26;
            function cy(v) { return cBot - v * (cBot - cTop); }
            ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, cBot); ctx.lineTo(w - padR, cBot); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(padL, cy(1)); ctx.lineTo(w - padR, cy(1)); ctx.stroke();

            // shaded band between the step functions
            ctx.fillStyle = 'rgba(124,58,237,0.20)';
            for (i = 0; i < BINS; i++) {
                var yA = cy(cumP[i]), yB = cy(cumQ[i]);
                ctx.fillRect(bx(i), Math.min(yA, yB), bw, Math.abs(yA - yB));
            }
            function stair(cum, colour) {
                ctx.strokeStyle = colour; ctx.lineWidth = 2.2;
                ctx.beginPath();
                ctx.moveTo(padL, cy(0));
                for (var k = 0; k < BINS; k++) {
                    ctx.lineTo(bx(k), cy(k === 0 ? 0 : cum[k - 1]));
                    ctx.lineTo(bx(k), cy(cum[k]));
                    ctx.lineTo(bx(k) + bw, cy(cum[k]));
                }
                ctx.stroke();
            }
            stair(cumP, TEAL);
            stair(cumQ, ORANGE);
            ctx.fillStyle = '#7c3aed'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('shaded area between the cumulative curves  =  W₁', padL + 2, cTop - 6);
            ctx.fillStyle = '#aaa'; ctx.textAlign = 'right';
            ctx.fillText('1', padL - 4, cy(1) + 3);
            ctx.fillText('0', padL - 4, cy(0) + 3);

            r1.textContent = 'W₁(P, Q) = ' + W1.toFixed(4) +
                '        total variation = ' + TV.toFixed(4) +
                (TV > 0.995 ? '  (pinned — no overlap left)' : '');

            if (TV > 0.995) {
                verdict.textContent = '✗ The piles no longer overlap. TV has saturated at 1 and stopped responding; W₁ = ' +
                    W1.toFixed(2) + ' is still tracking the distance.';
                verdict.style.color = '#c0392b';
            } else if (W1 < 0.02) {
                verdict.textContent = '✓ Q sits on P — no dirt needs moving at all, so W₁ ≈ 0.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = 'Moving Q one position further costs about ' + W1.toFixed(2) +
                    ' units of work — and every bit of that is a usable gradient.';
                verdict.style.color = '#999';
            }
        }

        [sMu, sSig].forEach(function (s) { s.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 22 (Lecture 6): the ELBO gap is exactly KL(q || posterior)
    // ─────────────────────────────────────────────────────────
    function initElbo() {
        var root = byId('widget-elbo');
        if (!root) return;
        makeTitle(root, 'Try it — why the E-step makes the bound exact');

        var LOGPX = -2.10;                 // log p(x): fixed, unknown to the algorithm
        var POST_M = 0.9, POST_S = 0.75;   // the true posterior p(z|x)

        var sMu = slider(root, 'mean of q', -2.5, 3.5, 0.02, -1.1);
        var sSig = slider(root, 'std of q', 0.25, 2.5, 0.02, 1.7);
        var bSnap = make('button', null, make('div', null, root), 'Run the E-step  (set q = p(z|x))');

        var canvas = makeCanvas(root, 250);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Left: the true posterior p(z|x) in teal, fixed, and your approximation q in orange. Right: the identity that ' +
            'runs the whole algorithm, drawn as a bar. The total height log p(x) never moves — it is a property of the ' +
            'data and the model, not of q. What q controls is how that fixed height is split between the part you can ' +
            'compute, the ELBO, and the part you waste, which is exactly D_KL(q ‖ p(z|x)). Since a KL is never negative, ' +
            'the ELBO can never exceed log p(x), which is why it is a lower bound at all. Drag q away from the posterior ' +
            'and the purple waste grows; drag it on top and the waste vanishes and the bound touches. That is precisely ' +
            'what the E-step does — and it is also exactly what a VAE cannot do, because for a deep decoder the posterior ' +
            'has no closed form, so an encoder network has to guess at it and the purple sliver never quite closes.');

        function klGauss(m1, s1, m2, s2) {
            return Math.log(s2 / s1) + (s1 * s1 + (m1 - m2) * (m1 - m2)) / (2 * s2 * s2) - 0.5;
        }

        function draw() {
            var mq = +sMu.value, sq = +sSig.value;
            sMu._val.textContent = mq.toFixed(2);
            sSig._val.textContent = sq.toFixed(2);
            var kl = klGauss(mq, sq, POST_M, POST_S);
            var elbo = LOGPX - kl;

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var colW = w * 0.56;

            // ── left: the two densities ──
            var padL = 28, baseY = h - 34, topY = 26;
            var X0 = -3.5, X1 = 4.5, maxV = 0.62;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (colW - padL - 20); }
            function py(v) { return baseY - Math.min(v / maxV, 1) * (baseY - topY); }
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(colW - 20, baseY); ctx.stroke();
            var x;
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.02) {
                var y = py(gaussPdf(x, POST_M, POST_S));
                x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.4; ctx.stroke();
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.02) {
                y = py(gaussPdf(x, mq, sq));
                x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 2.4; ctx.setLineDash([6, 4]); ctx.stroke();
            ctx.setLineDash([]);
            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('true posterior p(z|x)', padL, 14);
            ctx.fillStyle = ORANGE; ctx.fillText('your q(z)', padL, 28);
            ctx.fillStyle = '#999'; ctx.textAlign = 'center';
            ctx.fillText('z', (padL + colW - 20) / 2, h - 12);

            // ── right: the decomposition bar ──
            var bx = colW + 40, bw = 82;
            var bTop = 40, bBot = h - 42;
            var total = 3.2;                       // display scale in nats
            function bh(v) { return (v / total) * (bBot - bTop); }
            var fullH = bh(total);
            var klH = Math.min(bh(kl), fullH);
            ctx.fillStyle = 'rgba(14,116,144,0.20)';
            ctx.fillRect(bx, bBot - (fullH - klH), bw, fullH - klH);
            ctx.strokeStyle = TEAL; ctx.lineWidth = 1.6;
            ctx.strokeRect(bx, bBot - (fullH - klH), bw, fullH - klH);
            ctx.fillStyle = 'rgba(124,58,237,0.26)';
            ctx.fillRect(bx, bBot - fullH, bw, klH);
            ctx.strokeStyle = '#7c3aed';
            ctx.strokeRect(bx, bBot - fullH, bw, klH);

            ctx.strokeStyle = '#111'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(bx - 12, bBot - fullH); ctx.lineTo(bx + bw + 12, bBot - fullH); ctx.stroke();
            ctx.fillStyle = '#111'; ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('log p(x) — fixed', bx + bw + 16, bBot - fullH + 4);
            ctx.fillStyle = '#7c3aed';
            ctx.fillText('gap = D_KL(q ‖ p(z|x))', bx + bw + 16, bBot - fullH + klH / 2 + 4);
            ctx.fillStyle = TEAL;
            ctx.fillText('ELBO — what you optimise', bx + bw + 16, bBot - (fullH - klH) / 2 + 4);

            r1.textContent = 'log p(x) = ' + LOGPX.toFixed(3) +
                '     ELBO = ' + elbo.toFixed(3) +
                '     gap = D_KL = ' + kl.toFixed(3);

            if (kl < 0.005) {
                verdict.textContent = '✓ q equals the posterior: the gap is 0 and the ELBO is exactly log p(x). The bound is tight.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = 'Wasting ' + kl.toFixed(3) + ' nats. The ELBO understates log p(x) by exactly that much — never more, never less.';
                verdict.style.color = '#999';
            }
        }

        bSnap.addEventListener('click', function () {
            sMu.value = POST_M; sSig.value = POST_S; draw();
        });
        [sMu, sSig].forEach(function (s) { s.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 23 (Lecture 6): EM on a 1-D Gaussian mixture
    // ─────────────────────────────────────────────────────────
    function initEm() {
        var root = byId('widget-em');
        if (!root) return;
        makeTitle(root, 'Try it — run EM by hand and watch the likelihood only ever go up');

        var DATA = (function () {
            var seed = 20260803, out = [];
            function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
            function randn() { return Math.sqrt(-2 * Math.log(rnd() || 1e-9)) * Math.cos(2 * Math.PI * rnd()); }
            var i;
            for (i = 0; i < 90; i++) out.push(-2.2 + 0.55 * randn());
            for (i = 0; i < 70; i++) out.push(0.9 + 0.42 * randn());
            for (i = 0; i < 60; i++) out.push(3.4 + 0.7 * randn());
            return out;
        })();

        var K = 3, comps = [], hist = [], phase = 'E';

        var sK = slider(root, 'number of components K', 1, 4, 1, 3);
        var controls = make('div', null, root);
        var bStep = make('button', null, controls, 'Run E-step');
        var bRun = make('button', 'secondary', controls, '▶ Run to convergence');
        var bReset = make('button', 'secondary', controls, 'Reset');

        var canvas = makeCanvas(root, 260);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Grey bars are 220 data points drawn from a three-component mixture. The teal curves are the components the ' +
            'model currently believes in, and the thick curve is their weighted sum. Step through by hand and watch what ' +
            'each half does. The E-step changes no parameters at all — it only recomputes, for every point, how much each ' +
            'component is responsible for it, which is the exact posterior P(z|x) available in closed form by Bayes\' rule. ' +
            'The M-step then treats those responsibilities as soft labels and refits each component to the points it owns, ' +
            'which is a weighted mean and variance and nothing harder. The number to watch is the log-likelihood trace in ' +
            'the corner: it is monotone. It never dips, not once, because the E-step makes the bound touch the likelihood ' +
            'and the M-step then climbs the bound — so the likelihood cannot fall below where it already was. Set K = 1 ' +
            'and the single Gaussian smears across all three clumps, which is the same capacity argument as mode collapse.');

        function reset() {
            K = +sK.value; comps = []; hist = []; phase = 'E';
            var lo = -3.6, hi = 4.8;
            for (var j = 0; j < K; j++) {
                comps.push({ a: 1 / K, mu: lo + (j + 0.5) * (hi - lo) / K + (j % 2 ? 0.4 : -0.4), s: 1.3 });
            }
            bStep.textContent = 'Run E-step';
        }
        reset();

        function resp() {
            var R = [], i, j;
            for (i = 0; i < DATA.length; i++) {
                var row = [], tot = 0;
                for (j = 0; j < K; j++) {
                    var v = comps[j].a * gaussPdf(DATA[i], comps[j].mu, comps[j].s);
                    row.push(v); tot += v;
                }
                for (j = 0; j < K; j++) row[j] = tot > 1e-300 ? row[j] / tot : 1 / K;
                R.push(row);
            }
            return R;
        }

        function logLik() {
            var ll = 0, i, j;
            for (i = 0; i < DATA.length; i++) {
                var t = 0;
                for (j = 0; j < K; j++) t += comps[j].a * gaussPdf(DATA[i], comps[j].mu, comps[j].s);
                ll += Math.log(Math.max(t, 1e-300));
            }
            return ll / DATA.length;
        }

        function mStep(R) {
            for (var j = 0; j < K; j++) {
                var n = 0, sm = 0, sv = 0, i;
                for (i = 0; i < DATA.length; i++) { n += R[i][j]; sm += R[i][j] * DATA[i]; }
                if (n < 1e-9) continue;
                var mu = sm / n;
                for (i = 0; i < DATA.length; i++) sv += R[i][j] * (DATA[i] - mu) * (DATA[i] - mu);
                comps[j].mu = mu;
                comps[j].s = Math.max(0.18, Math.sqrt(sv / n));
                comps[j].a = n / DATA.length;
            }
        }

        var lastR = null;
        function step() {
            if (phase === 'E') { lastR = resp(); phase = 'M'; bStep.textContent = 'Run M-step'; }
            else { mStep(lastR || resp()); phase = 'E'; bStep.textContent = 'Run E-step'; hist.push(logLik()); }
            draw();
        }

        function draw() {
            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var padL = 30, padR = 16, baseY = h - 30, topY = 22;
            var X0 = -4.2, X1 = 5.6, maxV = 0.42;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (w - padL - padR); }
            function py(v) { return baseY - Math.min(v / maxV, 1) * (baseY - topY); }

            // data histogram
            var NB = 46, counts = new Array(NB), i, j;
            for (i = 0; i < NB; i++) counts[i] = 0;
            DATA.forEach(function (d) {
                var b = Math.floor((d - X0) / (X1 - X0) * NB);
                if (b >= 0 && b < NB) counts[b]++;
            });
            var bw = (w - padL - padR) / NB;
            var dens = (X1 - X0) / NB * DATA.length;
            for (i = 0; i < NB; i++) {
                var v = counts[i] / dens;
                ctx.fillStyle = 'rgba(120,120,120,0.20)';
                ctx.fillRect(padL + i * bw, py(v), bw - 0.6, baseY - py(v));
            }
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();

            // components + mixture
            var x;
            for (j = 0; j < K; j++) {
                ctx.beginPath();
                for (x = X0; x <= X1; x += 0.02) {
                    var y = py(comps[j].a * gaussPdf(x, comps[j].mu, comps[j].s));
                    x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
                }
                ctx.strokeStyle = 'rgba(14,116,144,0.55)'; ctx.lineWidth = 1.5; ctx.stroke();
            }
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.02) {
                var t = 0;
                for (j = 0; j < K; j++) t += comps[j].a * gaussPdf(x, comps[j].mu, comps[j].s);
                var yy = py(t);
                x === X0 ? ctx.moveTo(px(x), yy) : ctx.lineTo(px(x), yy);
            }
            ctx.strokeStyle = ORANGE; ctx.lineWidth = 2.6; ctx.stroke();

            // likelihood trace
            if (hist.length > 1) {
                var tw = 118, th = 46, tx = w - padR - tw, ty = 28;
                ctx.fillStyle = 'rgba(255,255,255,0.9)';
                ctx.fillRect(tx, ty, tw, th);
                ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1; ctx.strokeRect(tx, ty, tw, th);
                var lo = Math.min.apply(null, hist), hi = Math.max.apply(null, hist);
                var rng = Math.max(hi - lo, 1e-6);
                ctx.beginPath();
                hist.forEach(function (v, k) {
                    var X = tx + 4 + k / Math.max(hist.length - 1, 1) * (tw - 8);
                    var Y = ty + th - 5 - (v - lo) / rng * (th - 12);
                    k === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
                });
                ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 1.8; ctx.stroke();
                ctx.fillStyle = '#7c3aed'; ctx.font = '8.6px Inter, sans-serif'; ctx.textAlign = 'left';
                ctx.fillText('log-likelihood', tx + 4, ty - 3);
            }

            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = '#888'; ctx.fillText('data', padL, 14);
            ctx.fillStyle = ORANGE; ctx.fillText('mixture', padL + 42, 14);
            ctx.fillStyle = 'rgba(14,116,144,0.8)'; ctx.fillText('components', padL + 106, 14);

            var ll = logLik();
            r1.textContent = 'K = ' + K + '    iterations = ' + hist.length +
                '    mean log-likelihood = ' + ll.toFixed(5) +
                '    next: ' + (phase === 'E' ? 'E-step' : 'M-step');

            var mono = true;
            for (var q = 1; q < hist.length; q++) if (hist[q] < hist[q - 1] - 1e-9) mono = false;
            if (hist.length < 2) {
                verdict.textContent = 'Step through an E and an M and the trace will start building.';
                verdict.style.color = '#999';
            } else if (mono) {
                verdict.textContent = '✓ ' + hist.length + ' iterations, not one decrease. EM cannot go downhill — that is the theorem, not luck.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = 'A decrease appeared — with this arithmetic that only happens from numerical underflow.';
                verdict.style.color = '#c0392b';
            }
        }

        var animId = null;
        function stop() { if (animId) { clearInterval(animId); animId = null; } bRun.textContent = '▶ Run to convergence'; }

        bStep.addEventListener('click', function () { stop(); step(); });
        bRun.addEventListener('click', function () {
            if (animId) { stop(); return; }
            bRun.textContent = '❚❚ Stop';
            animId = setInterval(function () {
                step();
                if (hist.length > 3 && Math.abs(hist[hist.length - 1] - hist[hist.length - 2]) < 1e-7) stop();
                if (hist.length > 60) stop();
            }, 260);
        });
        bReset.addEventListener('click', function () { stop(); reset(); draw(); });
        sK.addEventListener('input', function () { stop(); sK._val.textContent = sK.value; reset(); draw(); });

        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 24 (Lecture 7): score-function vs reparameterised gradients
    // Estimating d/dmu E_{z~N(mu,s)}[z^2], whose true value is 2*mu.
    // ─────────────────────────────────────────────────────────
    function initReparam() {
        var root = byId('widget-reparam');
        if (!root) return;
        makeTitle(root, 'Try it — the same gradient, estimated two ways');

        var sMu = slider(root, 'μ of q', -3, 3, 0.05, 1);
        var sSig = slider(root, 'σ of q', 0.2, 2, 0.05, 1);
        var sM = slider(root, 'samples per estimate, M', 1, 20, 1, 1);

        var canvas = makeCanvas(root, 250);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'A deliberately tiny problem where the answer is known exactly: estimate the gradient with respect to μ of ' +
            'the expectation of z² under z ~ 𝒩(μ, σ²). The true value is 2μ, marked by the black line. Both estimators ' +
            'below are unbiased — run either one enough times and its average lands on the black line — so the thing that ' +
            'separates them is not correctness but spread. The score-function estimator (also called REINFORCE, and the ' +
            'thing you are forced into if you refuse to reparameterise) multiplies the function value by the score, and ' +
            'the product swings wildly; the reparameterised estimator differentiates through z = μ + σ·ε and barely ' +
            'moves. Push σ down or μ away from zero and watch the orange histogram spread out while the teal one gets ' +
            'tighter — the analytic variances are μ⁴/σ² + 14μ² + 15σ² against a flat 4σ², so the gap grows without ' +
            'bound. That ratio is the whole reason VAEs are trainable and the naive approach is not.');

        var TRIALS = 4000;

        function randn() {
            var u = Math.random() || 1e-9, v = Math.random();
            return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        }

        function sample(mu, sg, M, reparam) {
            var acc = 0;
            for (var j = 0; j < M; j++) {
                var e = randn(), z = mu + sg * e;
                acc += reparam ? 2 * z : (z * z) * ((z - mu) / (sg * sg));
            }
            return acc / M;
        }

        function draw() {
            var mu = +sMu.value, sg = +sSig.value, M = +sM.value;
            sMu._val.textContent = mu.toFixed(2);
            sSig._val.textContent = sg.toFixed(2);
            sM._val.textContent = M;

            var truth = 2 * mu;
            var varR = 4 * sg * sg / M;
            var varS = (Math.pow(mu, 4) / (sg * sg) + 14 * mu * mu + 15 * sg * sg) / M;

            var R = [], S = [], i;
            for (i = 0; i < TRIALS; i++) { R.push(sample(mu, sg, M, true)); S.push(sample(mu, sg, M, false)); }

            function stats(a) {
                var m = 0, k;
                for (k = 0; k < a.length; k++) m += a[k];
                m /= a.length;
                var v = 0;
                for (k = 0; k < a.length; k++) v += (a[k] - m) * (a[k] - m);
                return { mean: m, sd: Math.sqrt(v / a.length) };
            }
            var st = stats(R), ss = stats(S);

            // shared axis: wide enough for the score estimator, capped so it stays readable
            var half = Math.min(Math.max(3.2 * Math.sqrt(varS), 1.2), 60);
            var X0 = truth - half, X1 = truth + half;

            var w = canvas._w, h = canvas._h;
            var padL = 22, padR = 16, baseY = h - 34, topY = 26;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (w - padL - padR); }
            ctx.clearRect(0, 0, w, h);

            var NB = 70;
            function hist(a) {
                var c = new Array(NB), k;
                for (k = 0; k < NB; k++) c[k] = 0;
                for (k = 0; k < a.length; k++) {
                    var b = Math.floor((a[k] - X0) / (X1 - X0) * NB);
                    if (b >= 0 && b < NB) c[b]++;
                }
                return c;
            }
            var hR = hist(R), hS = hist(S);
            var top = 0;
            for (i = 0; i < NB; i++) { top = Math.max(top, hR[i], hS[i]); }
            top = Math.max(top, 1);
            var bw = (w - padL - padR) / NB;

            for (i = 0; i < NB; i++) {
                var yS = (hS[i] / top) * (baseY - topY);
                ctx.fillStyle = 'rgba(217,119,6,0.45)';
                ctx.fillRect(padL + i * bw, baseY - yS, bw, yS);
            }
            for (i = 0; i < NB; i++) {
                var yR = (hR[i] / top) * (baseY - topY);
                ctx.fillStyle = 'rgba(14,116,144,0.60)';
                ctx.fillRect(padL + i * bw, baseY - yR, bw, yR);
            }

            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();
            ctx.strokeStyle = '#111'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(px(truth), topY - 8); ctx.lineTo(px(truth), baseY + 5); ctx.stroke();
            ctx.fillStyle = '#111'; ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('true gradient = ' + truth.toFixed(2), px(truth), topY - 12);

            ctx.textAlign = 'left'; ctx.font = '11px Inter, sans-serif';
            ctx.fillStyle = TEAL; ctx.fillText('reparameterised', padL + 4, 14);
            ctx.fillStyle = ORANGE; ctx.fillText('score function (REINFORCE)', padL + 108, 14);
            ctx.fillStyle = '#999'; ctx.font = '9.4px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('estimated gradient over ' + TRIALS + ' independent runs', w / 2, h - 12);

            var ratio = Math.sqrt(varS / varR);
            r1.textContent = 'reparameterised:  mean ' + st.mean.toFixed(3) + '   std ' + st.sd.toFixed(3) +
                '   (analytic ' + Math.sqrt(varR).toFixed(3) + ')';
            r2.textContent = 'score function:   mean ' + ss.mean.toFixed(3) + '   std ' + ss.sd.toFixed(3) +
                '   (analytic ' + Math.sqrt(varS).toFixed(3) + ')';

            verdict.textContent = 'Both are unbiased — both means sit on ' + truth.toFixed(2) +
                '. But the score-function estimator is ' + ratio.toFixed(1) +
                '× wider, so it needs about ' + Math.round(ratio * ratio) +
                '× as many samples for the same precision.';
            verdict.style.color = ratio > 3 ? '#c0392b' : '#0e7490';
        }

        [sMu, sSig, sM].forEach(function (s) { s.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 25 (Lecture 7): the two ELBO terms pulling against each other
    // Linear decoder, so the optimum is closed form: mu = 2x/(2+b), s^2 = b/(2+b)
    // ─────────────────────────────────────────────────────────
    function initVaeLatent() {
        var root = byId('widget-vaelatent');
        if (!root) return;
        makeTitle(root, 'Try it — reconstruction against regularisation');

        var XS = [-2.4, -1.5, -0.7, 0.1, 0.9, 1.7, 2.5];
        var sB = slider(root, 'weight on the KL term, β', 0, 12, 0.05, 1);

        var canvas = makeCanvas(root, 260);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Seven data points, a linear decoder, and a Gaussian encoder — small enough that the optimum of the ELBO has ' +
            'a closed form, so nothing here is being trained or approximated. Each coloured bump is one point\'s posterior ' +
            'q(z|x); the dashed curve is the prior. At β = 0 the KL term is switched off and the optimum is μ = x with ' +
            'σ → 0: seven sharp spikes, perfect reconstruction, and an ordinary autoencoder. It also has enormous gaps ' +
            'between the spikes, and decoding a z from one of those gaps gives nonsense — which is exactly why a plain ' +
            'autoencoder cannot generate. Turn β up and the bumps widen and slide toward zero, filling the space in, until ' +
            'at large β they all sit on top of the prior and every input has the same code. That is posterior collapse: ' +
            'the latent carries no information and the decoder ignores it. Somewhere in between the space is both ' +
            'informative and gap-free, which is the only regime where sampling z ~ 𝒩(0,1) and decoding produces anything.');

        function draw() {
            var b = +sB.value;
            sB._val.textContent = b.toFixed(2);
            // exact optimum of  -(mu-x)^2 - s^2 - (b/2)(s^2 + mu^2 - 1 - 2 log s)
            var shrink = 2 / (2 + b);
            var sg = Math.sqrt(Math.max(b / (2 + b), 1e-6));

            var recon = 0, kl = 0, i;
            for (i = 0; i < XS.length; i++) {
                var mu = shrink * XS[i];
                recon += (mu - XS[i]) * (mu - XS[i]) + sg * sg;
                kl += 0.5 * (sg * sg + mu * mu - 1 - 2 * Math.log(sg));
            }
            recon /= XS.length; kl /= XS.length;

            var w = canvas._w, h = canvas._h;
            var padL = 26, padR = 16, baseY = h - 52, topY = 26;
            var X0 = -4, X1 = 4, maxV = 1.5;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (w - padL - padR); }
            function py(v) { return baseY - Math.min(v / maxV, 1) * (baseY - topY); }
            ctx.clearRect(0, 0, w, h);

            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();

            // prior
            var x;
            ctx.beginPath();
            for (x = X0; x <= X1; x += 0.02) {
                var yp = py(gaussPdf(x, 0, 1));
                x === X0 ? ctx.moveTo(px(x), yp) : ctx.lineTo(px(x), yp);
            }
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]); ctx.stroke();
            ctx.setLineDash([]);

            // one posterior per data point
            var cols = ['#0e7490', '#d97706', '#7c3aed', '#c0392b', '#0891b2', '#a16207', '#4338ca'];
            for (i = 0; i < XS.length; i++) {
                var mu = shrink * XS[i];
                ctx.beginPath();
                for (x = X0; x <= X1; x += 0.02) {
                    var yy = py(gaussPdf(x, mu, sg));
                    x === X0 ? ctx.moveTo(px(x), yy) : ctx.lineTo(px(x), yy);
                }
                ctx.strokeStyle = cols[i]; ctx.lineWidth = 1.8; ctx.stroke();
                ctx.fillStyle = cols[i];
                ctx.beginPath(); ctx.arc(px(mu), baseY + 10, 3.4, 0, 7); ctx.fill();
            }

            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = '#999'; ctx.fillText('prior p(z) = 𝒩(0,1)', padL + 4, 14);
            ctx.fillStyle = '#333'; ctx.fillText('each curve is one q(z|xᵢ)', padL + 132, 14);
            ctx.textAlign = 'center'; ctx.fillStyle = '#aaa'; ctx.font = '9.4px Inter, sans-serif';
            ctx.fillText('latent axis z  ·  dots are the posterior means', w / 2, baseY + 32);

            r1.textContent = 'μ = ' + shrink.toFixed(3) + '·x     σ = ' + sg.toFixed(3);
            r2.textContent = 'reconstruction error = ' + recon.toFixed(4) +
                '     mean KL = ' + kl.toFixed(4);

            if (b < 0.03) {
                verdict.textContent = '✗ β = 0: a plain autoencoder. Reconstruction is perfect and the latent space is nothing but spikes and gaps — you cannot sample from it.';
                verdict.style.color = '#c0392b';
            } else if (shrink < 0.12) {
                verdict.textContent = '✗ Posterior collapse: every input now maps to essentially the prior, so z carries no information about x at all.';
                verdict.style.color = '#c0392b';
            } else {
                verdict.textContent = '✓ Codes still distinguishable (μ keeps ' + (shrink * 100).toFixed(0) +
                    '% of x) and the posteriors overlap enough to fill the space — this is the usable regime.';
                verdict.style.color = '#0e7490';
            }
        }

        sB.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 26 (Lecture 8): does a prior sample land anywhere the decoder knows?
    // Closed-form beta optimum again: mu = 2x/(2+b), sigma^2 = b/(2+b).
    // ─────────────────────────────────────────────────────────
    function initPriorCover() {
        var root = byId('widget-priorcover');
        if (!root) return;
        makeTitle(root, 'Try it — why sampling from the prior works at all');

        var XS = [[-2.2, 1.4], [1.8, 2.0], [-1.6, -1.9], [2.3, -1.2], [0.2, 2.6], [-2.8, 0.1], [1.1, -2.4], [0.6, 0.4]];
        var sB = slider(root, 'weight on the KL term, β', 0, 12, 0.05, 0);

        var canvas = makeCanvas(root, 300);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Eight data points encoded into a two-dimensional latent space, with the same closed-form optimum as the ' +
            'β widget in Ch.7 — teal circles are each point\'s posterior at two standard deviations, the dashed ellipse ' +
            'is the prior. The purple dots are fresh draws from 𝒩(0, I), exactly what you do at generation time, and ' +
            'each is marked filled if it lands inside a region some posterior actually covers and hollow if it does not. ' +
            'Start at β = 0 and essentially every draw is hollow: the posteriors have collapsed to points, so the prior ' +
            'samples land in territory the decoder was never trained on and would decode to noise. That is the precise ' +
            'sense in which a plain autoencoder cannot generate. Raise β and the circles inflate and slide inward until ' +
            'they tile the prior\'s bulk and the hit rate goes past 95%. Keep going and it stays high — but by then the ' +
            'posteriors have merged into one blob and the codes no longer say anything about which input they came from, ' +
            'which is the other failure. The KL term is buying exactly this coverage, and nothing else.');

        function randn() {
            var u = Math.random() || 1e-9, v = Math.random();
            return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        }

        var probes = [];
        for (var i = 0; i < 220; i++) probes.push([randn(), randn()]);

        function draw() {
            var b = +sB.value;
            sB._val.textContent = b.toFixed(2);
            var shrink = 2 / (2 + b);
            var sg = Math.sqrt(Math.max(b / (2 + b), 1e-9));

            var w = canvas._w, h = canvas._h;
            var cx = w / 2, cy = h / 2, scale = Math.min(w, h) / 9.2;
            function px(x) { return cx + x * scale; }
            function py(y) { return cy - y * scale; }

            ctx.clearRect(0, 0, w, h);
            // axes
            ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(px(-4.5), py(0)); ctx.lineTo(px(4.5), py(0)); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(px(0), py(-3.2)); ctx.lineTo(px(0), py(3.2)); ctx.stroke();

            // prior at 2 sd
            ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
            ctx.beginPath(); ctx.arc(px(0), py(0), 2 * scale, 0, 7); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#7c3aed'; ctx.font = '9.6px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('prior 𝒩(0, I), 2σ', px(0) + 6, py(2) - 6);

            // posteriors
            var mus = XS.map(function (x) { return [shrink * x[0], shrink * x[1]]; });
            mus.forEach(function (m) {
                ctx.fillStyle = 'rgba(14,116,144,0.16)';
                ctx.beginPath(); ctx.arc(px(m[0]), py(m[1]), Math.max(2 * sg * scale, 1.5), 0, 7); ctx.fill();
                ctx.strokeStyle = TEAL; ctx.lineWidth = 1.4;
                ctx.beginPath(); ctx.arc(px(m[0]), py(m[1]), Math.max(2 * sg * scale, 1.5), 0, 7); ctx.stroke();
                ctx.fillStyle = TEAL;
                ctx.beginPath(); ctx.arc(px(m[0]), py(m[1]), 2.6, 0, 7); ctx.fill();
            });

            // prior probes, filled if covered
            var hit = 0;
            probes.forEach(function (z) {
                var covered = false;
                for (var k = 0; k < mus.length; k++) {
                    var d = Math.hypot(z[0] - mus[k][0], z[1] - mus[k][1]);
                    if (d < 2 * sg) { covered = true; break; }
                }
                if (covered) hit++;
                ctx.beginPath(); ctx.arc(px(z[0]), py(z[1]), 2.6, 0, 7);
                if (covered) { ctx.fillStyle = 'rgba(124,58,237,0.85)'; ctx.fill(); }
                else { ctx.strokeStyle = 'rgba(192,57,43,0.75)'; ctx.lineWidth = 1.1; ctx.stroke(); }
            });

            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('posteriors q(z|xᵢ)', 10, 15);
            ctx.fillStyle = '#7c3aed'; ctx.fillText('prior draws — filled = decodable', 128, 15);

            var pct = 100 * hit / probes.length;
            r1.textContent = 'μ = ' + shrink.toFixed(3) + '·x     σ = ' + sg.toFixed(3) +
                '     prior draws landing in a covered region: ' + pct.toFixed(1) + '%';

            if (b < 0.03) {
                verdict.textContent = '✗ β = 0: the posteriors are points, so almost nothing is covered. Every generated sample would come from territory the decoder has never seen.';
                verdict.style.color = '#c0392b';
            } else if (pct < 70) {
                verdict.textContent = 'Only ' + pct.toFixed(0) + '% covered — most prior draws still land in voids between the encoded regions.';
                verdict.style.color = '#d97706';
            } else if (shrink < 0.2) {
                verdict.textContent = '⚠ ' + pct.toFixed(0) + '% covered, but the posteriors have merged: μ keeps only ' +
                    (shrink * 100).toFixed(0) + '% of x, so the code barely identifies its input. Coverage bought at the price of information.';
                verdict.style.color = '#d97706';
            } else {
                verdict.textContent = '✓ ' + pct.toFixed(0) + '% of prior draws are decodable and the codes still separate the inputs. This is the regime generation needs.';
                verdict.style.color = '#0e7490';
            }
        }

        sB.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 27 (Lecture 8): vector quantisation on a grid
    // ─────────────────────────────────────────────────────────
    function initVq() {
        var root = byId('widget-vq');
        if (!root) return;
        makeTitle(root, 'Try it — a tiny codebook, a grid, and an enormous number of pictures');

        var GRID = 8;
        var sM = slider(root, 'codebook size M', 2, 64, 1, 6);

        var canvas = makeCanvas(root, 250);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'The encoder does not squeeze an image into one vector — it produces a whole grid of them, one per spatial ' +
            'location, and each is quantised separately against the same codebook. Left is what the encoder emitted, ' +
            'right is the same grid after every cell has been snapped to its nearest codebook entry, with the palette ' +
            'shown between them. At M = 2 the result is posterised beyond recognition; by M = 16 the two panels are hard ' +
            'to tell apart. Now read the second line: even the crude settings represent an astronomical number of ' +
            'distinct configurations, because the count is M raised to the number of cells, not M. That is the answer to ' +
            'the obvious objection that a few hundred vectors cannot possibly describe ImageNet — they are not being ' +
            'asked to. A 32×32 grid with M = 512 gives 512¹⁰²⁴ possibilities, which is why the reconstructions come out ' +
            'sharp rather than posterised.');

        // a smooth two-channel "latent field" over the grid, in [0,1]^2
        function field(i, j) {
            var u = i / (GRID - 1), v = j / (GRID - 1);
            return [
                0.5 + 0.42 * Math.sin(3.1 * u + 1.1 * v),
                0.5 + 0.42 * Math.cos(1.7 * u - 2.6 * v)
            ];
        }
        function colour(vec, alpha) {
            var r = Math.round(40 + 180 * vec[0]);
            var g = Math.round(90 + 120 * (1 - Math.abs(vec[0] - vec[1])));
            var b = Math.round(60 + 180 * vec[1]);
            return 'rgba(' + r + ',' + g + ',' + b + ',' + (alpha === undefined ? 1 : alpha) + ')';
        }

        function codebook(M) {
            // k-means-ish: seed on a lattice over the range the field actually occupies, then Lloyd iterations
            var pts = [], i, j;
            for (i = 0; i < GRID; i++) for (j = 0; j < GRID; j++) pts.push(field(i, j));
            var cb = [], k;
            for (k = 0; k < M; k++) cb.push([pts[(k * 7 + 3) % pts.length][0], pts[(k * 11 + 5) % pts.length][1]]);
            for (var it = 0; it < 25; it++) {
                var sums = [], cnts = [];
                for (k = 0; k < M; k++) { sums.push([0, 0]); cnts.push(0); }
                pts.forEach(function (p) {
                    var best = 0, bd = Infinity;
                    for (k = 0; k < M; k++) {
                        var d = (p[0] - cb[k][0]) * (p[0] - cb[k][0]) + (p[1] - cb[k][1]) * (p[1] - cb[k][1]);
                        if (d < bd) { bd = d; best = k; }
                    }
                    sums[best][0] += p[0]; sums[best][1] += p[1]; cnts[best]++;
                });
                for (k = 0; k < M; k++) if (cnts[k] > 0) { cb[k] = [sums[k][0] / cnts[k], sums[k][1] / cnts[k]]; }
            }
            return cb;
        }

        function draw() {
            var M = +sM.value;
            sM._val.textContent = M;
            var cb = codebook(M);

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var cell = Math.min(17, (w - 150) / (2 * GRID));
            var side = cell * GRID;
            var y0 = 40;
            var xL = 18, xR = w - 18 - side;

            var err = 0, i, j, k;
            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillStyle = TEAL; ctx.fillText('z_e(x) — what the encoder emitted', xL + side / 2, 22);
            ctx.fillStyle = '#7c3aed'; ctx.fillText('z_q(x) — after quantisation', xR + side / 2, 22);

            for (i = 0; i < GRID; i++) {
                for (j = 0; j < GRID; j++) {
                    var v = field(i, j);
                    ctx.fillStyle = colour(v);
                    ctx.fillRect(xL + j * cell, y0 + i * cell, cell, cell);

                    var best = 0, bd = Infinity;
                    for (k = 0; k < M; k++) {
                        var d = (v[0] - cb[k][0]) * (v[0] - cb[k][0]) + (v[1] - cb[k][1]) * (v[1] - cb[k][1]);
                        if (d < bd) { bd = d; best = k; }
                    }
                    err += Math.sqrt(bd);
                    ctx.fillStyle = colour(cb[best]);
                    ctx.fillRect(xR + j * cell, y0 + i * cell, cell, cell);
                }
            }
            err /= GRID * GRID;
            ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1;
            ctx.strokeRect(xL, y0, side, side);
            ctx.strokeRect(xR, y0, side, side);

            // palette between them
            var pxc = (xL + side + xR) / 2;
            ctx.fillStyle = '#666'; ctx.font = '9.4px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('codebook', pxc, y0 - 6);
            var sw = 13, cols = 3;
            for (k = 0; k < M; k++) {
                var cxp = pxc - (cols * sw) / 2 + (k % cols) * sw;
                var cyp = y0 + Math.floor(k / cols) * sw;
                if (cyp > y0 + side - sw) break;
                ctx.fillStyle = colour(cb[k]);
                ctx.fillRect(cxp, cyp, sw - 2, sw - 2);
            }
            if (M > cols * Math.floor(side / sw)) {
                ctx.fillStyle = '#999'; ctx.font = '8.6px Inter, sans-serif';
                ctx.fillText('+' + (M - cols * Math.floor(side / sw)) + ' more', pxc, y0 + side + 10);
            }

            var cells = GRID * GRID;
            var log10 = cells * Math.log10(M);
            r1.textContent = 'grid ' + GRID + '×' + GRID + ' = ' + cells + ' cells,  each snapped to one of ' + M +
                '     mean quantisation error = ' + err.toFixed(4);
            r2.textContent = 'distinct configurations: ' + M + '^' + cells + ' ≈ 10^' + log10.toFixed(0) +
                '   —  the capacity is Mᶜᵉˡˡˢ, not M';

            if (M <= 3) {
                verdict.textContent = '✗ Posterised — with ' + M + ' entries the grid cannot follow the field, and the error is ' + err.toFixed(3) + '.';
                verdict.style.color = '#c0392b';
            } else if (err < 0.045) {
                verdict.textContent = '✓ The two panels are now nearly indistinguishable, from a dictionary of just ' + M + ' vectors.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = 'Error ' + err.toFixed(3) + ' — visible banding remains. Add entries and watch it close.';
                verdict.style.color = '#999';
            }
        }

        sM.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Shared DDPM schedule: the standard linear beta from the paper.
    // ─────────────────────────────────────────────────────────
    var DDPM_T = 1000;
    var DDPM = (function () {
        var alpha = [0], abar = [1], ab = 1, t;
        for (t = 1; t <= DDPM_T; t++) {
            var beta = 1e-4 + (0.02 - 1e-4) * (t - 1) / (DDPM_T - 1);
            alpha[t] = 1 - beta;
            ab *= alpha[t];
            abar[t] = ab;
        }
        return { alpha: alpha, abar: abar, beta: function (t) { return 1 - alpha[t]; } };
    })();

    function nrand() {
        var u = Math.random() || 1e-9, v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    // ─────────────────────────────────────────────────────────
    // Widget 28 (Lecture 9): the forward chain, and the jump that skips it
    // ─────────────────────────────────────────────────────────
    function initForward() {
        var root = byId('widget-forward');
        if (!root) return;
        makeTitle(root, 'Try it — destroy an image on a schedule, then skip straight to any step');

        var G = 16;
        var IMG = (function () {
            var a = [], i, j;
            for (i = 0; i < G; i++) {
                a.push([]);
                for (j = 0; j < G; j++) {
                    var u = i / (G - 1) - 0.5, v = j / (G - 1) - 0.5;
                    var r = Math.sqrt(u * u + v * v);
                    // a soft disc with a bar across it — recognisable at a glance
                    var val = (r < 0.34 ? 1 : 0.08) + (Math.abs(v) < 0.07 && r < 0.45 ? -0.55 : 0);
                    a[i].push(Math.max(0, Math.min(1, val)) * 2 - 1);   // to roughly [-1,1]
                }
            }
            return a;
        })();

        var sT = slider(root, 'timestep t', 0, DDPM_T, 1, 0);
        var canvas = makeCanvas(root, 260);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'The forward process is fixed — no network, no learning, just a schedule of how much signal to keep and how ' +
            'much noise to add. The left panel applies the rule one step at a time, all t of them; the right panel uses ' +
            'the closed form and jumps there in a single shot. They agree because the composition of all those Gaussian ' +
            'steps collapses analytically, which is the result that makes training affordable: to train at step 900 you ' +
            'do not simulate 900 steps, you evaluate one formula. Watch the two coefficients underneath. The signal ' +
            'weight is the square root of ᾱ and the noise weight is the square root of one minus it, so their squares ' +
            'always sum to one and the variance never drifts — that is what the square roots in the transition are for. ' +
            'By t = 1000 the signal coefficient is under 0.01, which is the precise sense in which the image is gone and ' +
            'x_T is indistinguishable from a draw from the prior.');

        function draw() {
            var t = +sT.value;
            sT._val.textContent = t;
            var ab = t === 0 ? 1 : DDPM.abar[t];
            var sSig = Math.sqrt(ab), sNoise = Math.sqrt(1 - ab);

            // iterated chain, same noise seed each redraw so it is comparable
            var seed = 424242;
            function srand() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
            function srandn() { return Math.sqrt(-2 * Math.log(srand() || 1e-9)) * Math.cos(2 * Math.PI * srand()); }
            var iter = IMG.map(function (row) { return row.slice(); });
            var s, i, j;
            for (s = 1; s <= t; s++) {
                var a = DDPM.alpha[s], sa = Math.sqrt(a), sb = Math.sqrt(1 - a);
                for (i = 0; i < G; i++) for (j = 0; j < G; j++) iter[i][j] = sa * iter[i][j] + sb * srandn();
            }
            // closed form, independent noise
            var jump = IMG.map(function (row, ii) {
                return row.map(function (v) { return sSig * v + sNoise * srandn(); });
            });

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var cell = Math.min(11, (w - 200) / (2 * G));
            var side = cell * G, y0 = 34;
            var xA = 24, xB = xA + side + 40;

            function panel(x0, grid, label, col) {
                ctx.font = '10.4px Inter, sans-serif'; ctx.textAlign = 'center';
                ctx.fillStyle = col; ctx.fillText(label, x0 + side / 2, 22);
                for (var p = 0; p < G; p++) for (var q = 0; q < G; q++) {
                    var val = Math.max(-1, Math.min(1, grid[p][q]));
                    var g = Math.round((val + 1) / 2 * 255);
                    ctx.fillStyle = 'rgb(' + g + ',' + g + ',' + g + ')';
                    ctx.fillRect(x0 + q * cell, y0 + p * cell, cell, cell);
                }
                ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, side, side);
            }
            panel(xA, iter, 'iterated ' + t + ' step' + (t === 1 ? '' : 's'), TEAL);
            panel(xB, jump, 'one closed-form jump', '#7c3aed');

            // schedule plot
            var px0 = xB + side + 36, pw = w - px0 - 14, ph = 88, py0 = y0 + 6;
            if (pw > 60) {
                ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(px0, py0 + ph); ctx.lineTo(px0 + pw, py0 + ph); ctx.stroke();
                ctx.beginPath();
                for (var k = 1; k <= DDPM_T; k += 5) {
                    var X = px0 + (k / DDPM_T) * pw, Y = py0 + ph - DDPM.abar[k] * ph;
                    k === 1 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
                }
                ctx.strokeStyle = TEAL; ctx.lineWidth = 2; ctx.stroke();
                if (t > 0) {
                    ctx.strokeStyle = 'rgba(124,58,237,0.7)'; ctx.lineWidth = 1.3;
                    ctx.beginPath(); ctx.moveTo(px0 + (t / DDPM_T) * pw, py0); ctx.lineTo(px0 + (t / DDPM_T) * pw, py0 + ph); ctx.stroke();
                }
                ctx.fillStyle = TEAL; ctx.font = '9.4px Inter, sans-serif'; ctx.textAlign = 'left';
                ctx.fillText('ᾱ_t', px0 + 2, py0 - 4);
                ctx.fillStyle = '#aaa';
                ctx.fillText('0', px0 - 8, py0 + ph + 10);
                ctx.fillText('T', px0 + pw - 6, py0 + ph + 10);
            }

            r1.textContent = 'ᾱ_t = ' + ab.toExponential(3) +
                '     signal weight √ᾱ_t = ' + sSig.toFixed(4) +
                '     noise weight √(1−ᾱ_t) = ' + sNoise.toFixed(4);
            r2.textContent = 'sum of squares = ' + (sSig * sSig + sNoise * sNoise).toFixed(6) +
                '  — variance preserved exactly, at every t';

            if (t === 0) {
                verdict.textContent = 'At t = 0 nothing has happened yet: the closed form is the identity.';
                verdict.style.color = '#999';
            } else if (sSig < 0.02) {
                verdict.textContent = '✓ Signal weight is ' + sSig.toFixed(4) + ' — the image contributes essentially nothing and x_T is a draw from 𝒩(0, I).';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = 'Both panels are the same distribution — one took ' + t + ' steps, the other took one.';
                verdict.style.color = '#0e7490';
            }
        }

        sT.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 29 (Lecture 9): the true denoising posterior, checked against sampling
    // ─────────────────────────────────────────────────────────
    function initDdpmPost() {
        var root = byId('widget-ddpmpost');
        if (!root) return;
        makeTitle(root, 'Try it — the posterior we derived, against the one you get by sampling');

        var sT = slider(root, 'timestep t', 2, DDPM_T, 1, 60);
        var sX0 = slider(root, 'clean value x₀', -2, 2, 0.05, 1);
        var sEps = slider(root, 'noise draw ε that produced x_t', -2.5, 2.5, 0.05, 0.7);

        var canvas = makeCanvas(root, 240);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'One dimension, so everything is drawable. Given a clean value and the noise that corrupted it we know x_t ' +
            'exactly, and the question is where x_(t−1) must have been. The teal curve is the closed-form answer derived ' +
            'by completing the square in Bayes\' rule; the orange histogram is the empirical answer, obtained by drawing ' +
            'candidates from q(x_(t−1)|x₀) and weighting each by how plausibly it could have produced the x_t we ' +
            'observed. They coincide, which is the derivation confirmed rather than asserted. Two things to notice as ' +
            'you move t. The posterior is far narrower than the marginal it came from — knowing both endpoints pins the ' +
            'intermediate step down tightly, and that is exactly why conditioning on x₀ made the objective tractable. ' +
            'And the mean sits between x_t and the clean value, sliding toward x_t as t grows, which is precisely what ' +
            'the two coefficients in μ_q say it should do.');

        function draw() {
            var t = Math.round(+sT.value), x0 = +sX0.value, eps = +sEps.value;
            sT._val.textContent = t;
            sX0._val.textContent = x0.toFixed(2);
            sEps._val.textContent = eps.toFixed(2);

            var abT = DDPM.abar[t], abP = DDPM.abar[t - 1], aT = DDPM.alpha[t], bT = DDPM.beta(t);
            var xt = Math.sqrt(abT) * x0 + Math.sqrt(1 - abT) * eps;

            var btil = (1 - abP) / (1 - abT) * bT;
            var mu = Math.sqrt(aT) * (1 - abP) / (1 - abT) * xt + Math.sqrt(abP) * (1 - aT) / (1 - abT) * x0;
            var muEps = (1 / Math.sqrt(aT)) * (xt - (1 - aT) / Math.sqrt(1 - abT) * eps);
            var sd = Math.sqrt(btil);

            // importance-weighted empirical posterior
            var N = 60000, lo = mu - 5 * sd, hi = mu + 5 * sd, NB = 64;
            var hist = new Array(NB), i;
            for (i = 0; i < NB; i++) hist[i] = 0;
            var sw = 0, sx = 0, sxx = 0;
            for (i = 0; i < N; i++) {
                var xm1 = Math.sqrt(abP) * x0 + Math.sqrt(1 - abP) * nrand();
                var d = xt - Math.sqrt(aT) * xm1;
                var wgt = Math.exp(-d * d / (2 * bT));
                sw += wgt; sx += wgt * xm1; sxx += wgt * xm1 * xm1;
                var b = Math.floor((xm1 - lo) / (hi - lo) * NB);
                if (b >= 0 && b < NB) hist[b] += wgt;
            }
            var mMC = sx / sw, vMC = sxx / sw - mMC * mMC;

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var padL = 30, padR = 16, baseY = h - 34, topY = 24;
            function px(x) { return padL + (x - lo) / (hi - lo) * (w - padL - padR); }
            var hmax = 0;
            for (i = 0; i < NB; i++) hmax = Math.max(hmax, hist[i]);
            var bw = (w - padL - padR) / NB;
            for (i = 0; i < NB; i++) {
                var hh = (hist[i] / hmax) * (baseY - topY);
                ctx.fillStyle = 'rgba(217,119,6,0.45)';
                ctx.fillRect(padL + i * bw, baseY - hh, bw, hh);
            }
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();

            ctx.beginPath();
            var peak = gaussPdf(mu, mu, sd);
            for (var xv = lo; xv <= hi; xv += (hi - lo) / 400) {
                var yv = baseY - (gaussPdf(xv, mu, sd) / peak) * (baseY - topY);
                xv === lo ? ctx.moveTo(px(xv), yv) : ctx.lineTo(px(xv), yv);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.4; ctx.stroke();

            [[xt, '#7c3aed', 'x_t'], [x0, '#111', 'x₀']].forEach(function (m) {
                if (m[0] < lo || m[0] > hi) return;
                ctx.strokeStyle = m[1]; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
                ctx.beginPath(); ctx.moveTo(px(m[0]), topY - 8); ctx.lineTo(px(m[0]), baseY); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = m[1]; ctx.font = '9.6px Inter, sans-serif'; ctx.textAlign = 'center';
                ctx.fillText(m[2], px(m[0]), topY - 11);
            });

            ctx.font = '10.4px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('analytic q(x_{t−1} | x_t, x₀)', padL + 2, 14);
            ctx.fillStyle = ORANGE; ctx.fillText('empirical, by weighted sampling', padL + 168, 14);

            r1.textContent = 'analytic:  μ_q = ' + mu.toFixed(5) + '   σ = ' + sd.toExponential(3) +
                '        empirical:  μ = ' + mMC.toFixed(5) + '   σ = ' + Math.sqrt(vMC).toExponential(3);
            r2.textContent = 'x_t = ' + xt.toFixed(4) + '     μ_q via the ε-form = ' + muEps.toFixed(5) +
                '   (identical to μ_q above — same quantity, two parameterisations)';

            var relErr = Math.abs(mu - mMC) / Math.max(Math.abs(mu), 1e-9);
            verdict.textContent = '✓ Analytic and empirical means agree to ' + (relErr * 100).toExponential(1) +
                '%, and the ε-form matches to ' + Math.abs(mu - muEps).toExponential(1) + '. The derivation holds.';
            verdict.style.color = '#0e7490';
        }

        [sT, sX0, sEps].forEach(function (s) { s.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 30 (Lecture 9): run the reverse chain with a perfect noise predictor
    // ─────────────────────────────────────────────────────────
    function initReverse() {
        var root = byId('widget-reverse');
        if (!root) return;
        makeTitle(root, 'Try it — noise in, data out, one denoising step at a time');

        // target: a 1-D mixture the chain has to rediscover
        function targetSample() {
            return Math.random() < 0.45 ? -1.5 + 0.28 * nrand() : 1.2 + 0.42 * nrand();
        }
        function targetPdf(x) { return 0.45 * gaussPdf(x, -1.5, 0.28) + 0.55 * gaussPdf(x, 1.2, 0.42); }

        var N = 900;
        var parts = [], step = DDPM_T, animId = null;

        var controls = make('div', null, root);
        var bPlay = make('button', null, controls, '▶ Run the reverse chain');
        var bReset = make('button', 'secondary', controls, 'Reset to noise');

        var canvas = makeCanvas(root, 250);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Nine hundred particles start as pure noise and are pushed backwards along the chain. The reverse step is ' +
            'exactly the one derived in the notes — take the current point, subtract the predicted noise scaled by the ' +
            'schedule, then add a little fresh noise of the posterior\'s variance — but instead of a trained U-Net the ' +
            'predictor here is the exact optimal one, computed analytically for this toy target. That isolates the ' +
            'question worth asking: assuming the network learns its job perfectly, does the sampling procedure actually ' +
            'reconstruct the distribution? It does. The histogram starts as a single Gaussian bump and splits into the ' +
            'two modes of the target, in the right proportions, purely by following the reverse transitions. Note how ' +
            'little happens for most of the chain and how much happens near the end — the schedule spends its early ' +
            'steps on coarse structure and its last steps on detail, which is why sample quality is so sensitive to the ' +
            'final few hundred steps.');

        function reset() {
            parts = [];
            for (var i = 0; i < N; i++) parts.push(nrand());
            step = DDPM_T;
        }
        reset();

        // exact score-based noise predictor for the toy target
        function epsHat(xt, t) {
            var ab = DDPM.abar[t], sa = Math.sqrt(ab), sv = 1 - ab;
            // q(x_t) = mixture of N(sa*mu_k, sa^2 sd_k^2 + sv); eps = -(sqrt(1-ab)) * score
            var comps = [[0.45, -1.5, 0.28], [0.55, 1.2, 0.42]];
            var num = 0, den = 0;
            comps.forEach(function (c) {
                var m = sa * c[1], v = ab * c[2] * c[2] + sv;
                var pdf = Math.exp(-0.5 * (xt - m) * (xt - m) / v) / Math.sqrt(2 * Math.PI * v);
                num += c[0] * pdf * (-(xt - m) / v);
                den += c[0] * pdf;
            });
            var score = den > 1e-300 ? num / den : 0;
            return -Math.sqrt(sv) * score;
        }

        function reverseStep() {
            if (step < 1) return;
            var t = step, a = DDPM.alpha[t], ab = DDPM.abar[t], b = DDPM.beta(t);
            var abP = t > 1 ? DDPM.abar[t - 1] : 1;
            var sd = t > 1 ? Math.sqrt((1 - abP) / (1 - ab) * b) : 0;
            for (var i = 0; i < N; i++) {
                var x = parts[i];
                var mu = (1 / Math.sqrt(a)) * (x - (1 - a) / Math.sqrt(1 - ab) * epsHat(x, t));
                parts[i] = mu + sd * nrand();
            }
            step--;
        }

        function draw() {
            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var padL = 24, padR = 16, baseY = h - 34, topY = 24;
            var X0 = -3.2, X1 = 3.2, NB = 60;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (w - padL - padR); }
            var hist = new Array(NB), i;
            for (i = 0; i < NB; i++) hist[i] = 0;
            parts.forEach(function (v) {
                var b = Math.floor((v - X0) / (X1 - X0) * NB);
                if (b >= 0 && b < NB) hist[b]++;
            });
            var dens = (X1 - X0) / NB * N;
            var maxV = 0.75, bw = (w - padL - padR) / NB;
            for (i = 0; i < NB; i++) {
                var d = hist[i] / dens;
                var hh = Math.min(d / maxV, 1) * (baseY - topY);
                ctx.fillStyle = 'rgba(217,119,6,0.45)';
                ctx.fillRect(padL + i * bw, baseY - hh, bw, hh);
            }
            ctx.beginPath();
            for (var xv = X0; xv <= X1; xv += 0.02) {
                var yv = baseY - Math.min(targetPdf(xv) / maxV, 1) * (baseY - topY);
                xv === X0 ? ctx.moveTo(px(xv), yv) : ctx.lineTo(px(xv), yv);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.4; ctx.stroke();
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();

            ctx.font = '10.4px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('target distribution', padL + 2, 14);
            ctx.fillStyle = ORANGE; ctx.fillText('the particles, right now', padL + 122, 14);

            r1.textContent = 'at t = ' + step + ' of ' + DDPM_T + (step === 0 ? '  — chain complete' : '');
            if (step === DDPM_T) {
                verdict.textContent = 'Pure noise. Nothing about the target is present yet.';
                verdict.style.color = '#999';
            } else if (step === 0) {
                verdict.textContent = '✓ Both modes recovered, in roughly the right proportions — from noise, using only the reverse transitions.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = 'Denoising… watch the single bump split as the last few hundred steps go by.';
                verdict.style.color = '#999';
            }
        }

        function stop() { if (animId) { cancelAnimationFrame(animId); animId = null; } bPlay.textContent = '▶ Run the reverse chain'; }

        bPlay.addEventListener('click', function () {
            if (animId) { stop(); return; }
            if (step < 1) { reset(); }
            bPlay.textContent = '❚❚ Pause';
            function loop() {
                for (var k = 0; k < 12 && step > 0; k++) reverseStep();
                draw();
                if (step > 0) { animId = requestAnimationFrame(loop); } else { stop(); }
            }
            animId = requestAnimationFrame(loop);
        });
        bReset.addEventListener('click', function () { stop(); reset(); draw(); });
        document.addEventListener('visibilitychange', function () { if (document.hidden) stop(); });

        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 31 (Lecture 10): sinusoidal time embedding
    // ─────────────────────────────────────────────────────────
    function initTimeEmbed() {
        var root = byId('widget-timeembed');
        if (!root) return;
        makeTitle(root, 'Try it — why a scalar t has to become a vector');

        var sD = slider(root, 'embedding dimension D', 8, 64, 2, 32);
        var sBase = slider(root, 'frequency base', 10, 100000, 10, 10000);

        var canvas = makeCanvas(root, 250);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Left: the embedding itself — every column is one timestep, every row one of the D coordinates, and the ' +
            'stripes are the sine and cosine pairs at geometrically spaced frequencies. The top rows oscillate quickly ' +
            'and resolve neighbouring timesteps; the bottom rows barely move across the whole range and encode roughly ' +
            'where in the schedule you are. Right: the similarity between every pair of timesteps, which is what the ' +
            'network actually experiences. A bright diagonal that fades smoothly outward is exactly what you want — ' +
            'nearby steps get similar conditioning, so the network can interpolate, while distant steps are clearly ' +
            'distinguishable. Drag the frequency base down to something small and watch the pattern become periodic: ' +
            'timesteps far apart start looking identical, and the network can no longer tell whether it is denoising a ' +
            'nearly clean image or pure noise. The geometric spacing is what covers every scale at once with D numbers.');

        function embed(t, D, base) {
            var v = [], i;
            for (i = 0; i < D / 2; i++) {
                var w = 1 / Math.pow(base, 2 * i / D);
                v.push(Math.sin(w * t));
                v.push(Math.cos(w * t));
            }
            return v;
        }

        function draw() {
            var D = Math.round(+sD.value / 2) * 2, base = +sBase.value;
            sD._val.textContent = D;
            sBase._val.textContent = base;
            var TT = 100, i, j;

            var embs = [];
            for (j = 0; j < TT; j++) embs.push(embed(j * 10, D, base));

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var gap = 40, panel = Math.min((w - gap - 30) / 2, h - 60);
            var x0 = 16, y0 = 34;

            // heat map of the embedding
            var cw = panel / TT, ch = panel / D;
            for (j = 0; j < TT; j++) for (i = 0; i < D; i++) {
                var val = (embs[j][i] + 1) / 2;
                var g = Math.round(255 * (1 - val));
                ctx.fillStyle = 'rgb(' + Math.round(14 + g * 0.75) + ',' + Math.round(116 + g * 0.5) + ',' + Math.round(144 + g * 0.42) + ')';
                ctx.fillRect(x0 + j * cw, y0 + i * ch, Math.ceil(cw), Math.ceil(ch));
            }
            ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, panel, panel);
            ctx.font = '10.4px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = TEAL;
            ctx.fillText('the embedding — D rows, t across', x0 + panel / 2, 22);
            ctx.fillStyle = '#aaa'; ctx.font = '9px Inter, sans-serif';
            ctx.fillText('t = 0', x0 + 14, y0 + panel + 12);
            ctx.fillText('t = T', x0 + panel - 14, y0 + panel + 12);

            // cosine similarity matrix
            var x1 = x0 + panel + gap;
            var sim = [];
            for (j = 0; j < TT; j++) {
                sim.push([]);
                for (i = 0; i < TT; i++) {
                    var dot = 0, na = 0, nb = 0;
                    for (var k = 0; k < D; k++) { dot += embs[j][k] * embs[i][k]; na += embs[j][k] * embs[j][k]; nb += embs[i][k] * embs[i][k]; }
                    sim[j].push(dot / Math.sqrt(na * nb));
                }
            }
            var sw2 = panel / TT;
            for (j = 0; j < TT; j++) for (i = 0; i < TT; i++) {
                var sv = (sim[j][i] + 1) / 2;
                var gg = Math.round(255 * (1 - sv));
                ctx.fillStyle = 'rgb(' + Math.round(124 + gg * 0.5) + ',' + Math.round(58 + gg * 0.75) + ',' + Math.round(237 + gg * 0.07) + ')';
                ctx.fillRect(x1 + i * sw2, y0 + j * sw2, Math.ceil(sw2), Math.ceil(sw2));
            }
            ctx.strokeStyle = '#ccc'; ctx.strokeRect(x1, y0, panel, panel);
            ctx.font = '10.4px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = '#7c3aed';
            ctx.fillText('similarity between timesteps', x1 + panel / 2, 22);

            // how far can two timesteps be and still look alike?
            var offDiag = 0, count = 0;
            for (j = 0; j < TT; j++) for (i = 0; i < TT; i++)
                if (Math.abs(i - j) > TT / 4) { offDiag += sim[j][i]; count++; }
            offDiag /= count;

            r1.textContent = 'D = ' + D + '   base = ' + base +
                '   mean similarity between far-apart timesteps: ' + offDiag.toFixed(3);
            if (offDiag > 0.5) {
                verdict.textContent = '✗ Distant timesteps still look ' + (offDiag * 100).toFixed(0) +
                    '% alike — the network cannot tell early from late, and conditioning is nearly useless.';
                verdict.style.color = '#c0392b';
            } else {
                verdict.textContent = '✓ A clear diagonal: neighbouring steps are similar enough to interpolate, distant ones clearly distinct.';
                verdict.style.color = '#0e7490';
            }
        }

        [sD, sBase].forEach(function (x) { x.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 32 (Lecture 10): classifier guidance
    // ─────────────────────────────────────────────────────────
    function initGuidance() {
        var root = byId('widget-guidance');
        if (!root) return;
        makeTitle(root, 'Try it — steer an unconditional model with a classifier gradient');

        // two classes; the unconditional model knows only the mixture
        var CLS = [{ w: 0.5, m: -1.5, s: 0.32 }, { w: 0.5, m: 1.5, s: 0.32 }];
        var sS = slider(root, 'guidance scale s', 0, 12, 0.1, 0);
        var N = 900, parts = [], step = DDPM_T, animId = null;

        var controls = make('div', null, root);
        var bPlay = make('button', null, controls, '▶ Sample');
        var bReset = make('button', 'secondary', controls, 'Reset');

        var canvas = makeCanvas(root, 240);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'The unconditional model has learned a two-mode distribution and has no idea the modes mean anything. A ' +
            'separate classifier, trained on noisy inputs, reports how likely each point is to be the right-hand class; ' +
            'its gradient with respect to the input is added to the score at every reverse step, scaled by s. At s = 0 ' +
            'you get the unconditional model back and both modes appear equally. Push s up and the left mode drains ' +
            'away — the guidance is bending every trajectory toward the region the classifier prefers. Keep pushing and ' +
            'the familiar failure appears: the samples pile into an unnaturally narrow spike, sharper than the true ' +
            'class-conditional distribution. Guidance does not sample from p(x|y); it samples from something ' +
            'proportional to p(x)p(y|x)^s, and only s = 1 is the honest posterior. Everything above that trades ' +
            'diversity for a classifier score, which is why real systems live around 3 to 8 and call it a taste knob.');

        function classMarg(x, t, k) {
            var ab = DDPM.abar[t], m = Math.sqrt(ab) * CLS[k].m, v = ab * CLS[k].s * CLS[k].s + (1 - ab);
            return Math.exp(-0.5 * (x - m) * (x - m) / v) / Math.sqrt(2 * Math.PI * v);
        }
        function uncondEps(x, t) {
            var ab = DDPM.abar[t], num = 0, den = 0;
            for (var k = 0; k < 2; k++) {
                var m = Math.sqrt(ab) * CLS[k].m, v = ab * CLS[k].s * CLS[k].s + (1 - ab);
                var pdf = CLS[k].w * classMarg(x, t, k);
                num += pdf * (-(x - m) / v); den += pdf;
            }
            return -Math.sqrt(1 - ab) * (num / den);
        }
        // d/dx log p(y = class 1 | x_t), computed exactly
        function clsGrad(x, t) {
            var eps = 1e-4;
            function lp(z) {
                var a = CLS[0].w * classMarg(z, t, 0), b = CLS[1].w * classMarg(z, t, 1);
                return Math.log(Math.max(b / (a + b), 1e-300));
            }
            return (lp(x + eps) - lp(x - eps)) / (2 * eps);
        }

        function reset() { parts = []; for (var i = 0; i < N; i++) parts.push(nrand()); step = DDPM_T; }
        reset();

        function stepOnce() {
            if (step < 1) return;
            var t = step, a = DDPM.alpha[t], ab = DDPM.abar[t], b = DDPM.beta(t);
            var abP = t > 1 ? DDPM.abar[t - 1] : 1;
            var sd = t > 1 ? Math.sqrt((1 - abP) / (1 - ab) * b) : 0;
            var sc = +sS.value;
            for (var i = 0; i < N; i++) {
                var x = parts[i];
                var e = uncondEps(x, t) - sc * Math.sqrt(1 - ab) * clsGrad(x, t);
                parts[i] = (1 / Math.sqrt(a)) * (x - (1 - a) / Math.sqrt(1 - ab) * e) + sd * nrand();
            }
            step--;
        }

        function draw() {
            sS._val.textContent = (+sS.value).toFixed(1);
            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var padL = 24, padR = 16, baseY = h - 34, topY = 24;
            var X0 = -3.4, X1 = 3.4, NB = 62;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (w - padL - padR); }
            var hist = new Array(NB), i;
            for (i = 0; i < NB; i++) hist[i] = 0;
            parts.forEach(function (v) { var b = Math.floor((v - X0) / (X1 - X0) * NB); if (b >= 0 && b < NB) hist[b]++; });
            var dens = (X1 - X0) / NB * N, maxV = 1.3, bw = (w - padL - padR) / NB;
            for (i = 0; i < NB; i++) {
                var hh = Math.min((hist[i] / dens) / maxV, 1) * (baseY - topY);
                ctx.fillStyle = 'rgba(217,119,6,0.5)';
                ctx.fillRect(padL + i * bw, baseY - hh, bw, hh);
            }
            [[0, 'rgba(120,120,120,0.55)'], [1, TEAL]].forEach(function (c) {
                ctx.beginPath();
                for (var xv = X0; xv <= X1; xv += 0.02) {
                    var d = gaussPdf(xv, CLS[c[0]].m, CLS[c[0]].s);
                    var yv = baseY - Math.min(d / maxV, 1) * (baseY - topY);
                    xv === X0 ? ctx.moveTo(px(xv), yv) : ctx.lineTo(px(xv), yv);
                }
                ctx.strokeStyle = c[1]; ctx.lineWidth = 2; ctx.stroke();
            });
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();
            ctx.font = '10.4px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('target class', padL + 2, 14);
            ctx.fillStyle = '#999'; ctx.fillText('other class', padL + 84, 14);
            ctx.fillStyle = ORANGE; ctx.fillText('samples', padL + 168, 14);

            var right = parts.filter(function (v) { return v > 0; }).length / N;
            var mean = parts.reduce(function (s2, v) { return s2 + v; }, 0) / N;
            var sd2 = Math.sqrt(parts.reduce(function (s2, v) { return s2 + (v - mean) * (v - mean); }, 0) / N);
            r1.textContent = 'at t = ' + step + '    in the target class: ' + (right * 100).toFixed(1) +
                '%    sample std: ' + sd2.toFixed(3) + '   (true class std is 0.320)';
            if (step > 0) { verdict.textContent = 'Sampling…'; verdict.style.color = '#999'; }
            else if (+sS.value < 0.05) {
                verdict.textContent = 'Unconditional: both modes, about half each. The model has no idea which class you wanted.';
                verdict.style.color = '#999';
            } else if (sd2 < 0.26) {
                verdict.textContent = '⚠ Over-guided — ' + (right * 100).toFixed(0) + '% on target, but the spread has collapsed to ' +
                    sd2.toFixed(2) + ' against a true 0.32. Diversity traded for classifier score.';
                verdict.style.color = '#d97706';
            } else {
                verdict.textContent = '✓ ' + (right * 100).toFixed(0) + '% on target with the spread roughly intact.';
                verdict.style.color = '#0e7490';
            }
        }

        function stop() { if (animId) { cancelAnimationFrame(animId); animId = null; } bPlay.textContent = '▶ Sample'; }
        bPlay.addEventListener('click', function () {
            if (animId) { stop(); return; }
            reset(); bPlay.textContent = '❚❚ Pause';
            function loop() {
                for (var k = 0; k < 14 && step > 0; k++) stepOnce();
                draw();
                if (step > 0) { animId = requestAnimationFrame(loop); } else { stop(); }
            }
            animId = requestAnimationFrame(loop);
        });
        bReset.addEventListener('click', function () { stop(); reset(); draw(); });
        sS.addEventListener('input', draw);
        document.addEventListener('visibilitychange', function () { if (document.hidden) stop(); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 33 (Lecture 10): DDIM — skipping steps, and determinism
    // ─────────────────────────────────────────────────────────
    function initDdim() {
        var root = byId('widget-ddim');
        if (!root) return;
        makeTitle(root, 'Try it — the same trained model, run two different ways');

        var comps = [[0.45, -1.5, 0.28], [0.55, 1.2, 0.42]];
        function targetPdf(x) { return 0.45 * gaussPdf(x, -1.5, 0.28) + 0.55 * gaussPdf(x, 1.2, 0.42); }
        function epsHat(xt, t) {
            var ab = DDPM.abar[t], sa = Math.sqrt(ab), sv = 1 - ab, num = 0, den = 0;
            comps.forEach(function (c) {
                var m = sa * c[1], v = ab * c[2] * c[2] + sv;
                var pdf = c[0] * Math.exp(-0.5 * (xt - m) * (xt - m) / v) / Math.sqrt(2 * Math.PI * v);
                num += pdf * (-(xt - m) / v); den += pdf;
            });
            return -Math.sqrt(sv) * (num / den);
        }

        var sS = slider(root, 'sampling steps S', 5, 200, 5, 20);
        var controls = make('div', null, root);
        var bMode = make('button', null, controls, 'σ = 0  (DDIM, deterministic)');
        var bRun = make('button', 'secondary', controls, 'Sample again');
        var stochastic = false;

        var canvas = makeCanvas(root, 240);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var r2 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'One trained noise predictor, sampled two ways. The steps slider picks how many of the 1000 timesteps the ' +
            'sampler actually visits — DDPM has to visit all of them because its forward chain is Markovian, while DDIM ' +
            'may jump along any sub-sequence because the objective only ever constrained the marginals. Drop the count ' +
            'to 20 and the deterministic sampler still reproduces both modes at roughly the right weights, which is the ' +
            'fifty-fold speed-up in one picture. The button switches σ between zero and the value that reproduces DDPM ' +
            'exactly; watch the reported spread of repeated runs from one fixed starting point, which is zero for σ = 0 ' +
            'and non-zero otherwise. That is the whole of invertibility: with no noise injected, one x_T corresponds to ' +
            'one x₀, so you can run the map backwards and edit a real image in latent space.');

        function sample(S, stoch, seedParts) {
            var ts = [], i;
            for (i = 1; i <= S; i++) ts.push(Math.max(1, Math.round(i * DDPM_T / S)));
            var out = seedParts.slice();
            for (i = ts.length - 1; i >= 0; i--) {
                var tc = ts[i], tp = i === 0 ? 0 : ts[i - 1];
                var abC = DDPM.abar[tc], abP = tp === 0 ? 1 : DDPM.abar[tp];
                var sig = 0;
                if (stoch) sig = Math.sqrt((1 - abP) / (1 - abC)) * Math.sqrt(Math.max(1 - abC / abP, 0));
                var inner = Math.max(1 - abP - sig * sig, 0);
                for (var k = 0; k < out.length; k++) {
                    var x = out[k], e = epsHat(x, tc);
                    var pred = (x - Math.sqrt(1 - abC) * e) / Math.sqrt(abC);
                    out[k] = Math.sqrt(abP) * pred + Math.sqrt(inner) * e + sig * nrand();
                }
            }
            return out;
        }

        var N = 900, seeds = [];
        function reseed() { seeds = []; for (var i = 0; i < N; i++) seeds.push(nrand()); }
        reseed();

        function draw() {
            var S = +sS.value;
            sS._val.textContent = S;
            var out = sample(S, stochastic, seeds);

            // determinism probe: same single start, run 8 times
            var probe = [];
            for (var r = 0; r < 8; r++) probe.push(sample(S, stochastic, [1.234])[0]);
            var pm = probe.reduce(function (a, b) { return a + b; }, 0) / probe.length;
            var psd = Math.sqrt(probe.reduce(function (a, b) { return a + (b - pm) * (b - pm); }, 0) / probe.length);

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var padL = 24, padR = 16, baseY = h - 34, topY = 24;
            var X0 = -3.2, X1 = 3.2, NB = 60;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (w - padL - padR); }
            var hist = new Array(NB), i;
            for (i = 0; i < NB; i++) hist[i] = 0;
            out.forEach(function (v) { var b = Math.floor((v - X0) / (X1 - X0) * NB); if (b >= 0 && b < NB) hist[b]++; });
            var dens = (X1 - X0) / NB * N, maxV = 0.8, bw = (w - padL - padR) / NB;
            for (i = 0; i < NB; i++) {
                var hh = Math.min((hist[i] / dens) / maxV, 1) * (baseY - topY);
                ctx.fillStyle = 'rgba(217,119,6,0.45)';
                ctx.fillRect(padL + i * bw, baseY - hh, bw, hh);
            }
            ctx.beginPath();
            for (var xv = X0; xv <= X1; xv += 0.02) {
                var yv = baseY - Math.min(targetPdf(xv) / maxV, 1) * (baseY - topY);
                xv === X0 ? ctx.moveTo(px(xv), yv) : ctx.lineTo(px(xv), yv);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.4; ctx.stroke();
            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();
            ctx.font = '10.4px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = TEAL; ctx.fillText('target', padL + 2, 14);
            ctx.fillStyle = ORANGE; ctx.fillText(S + ' sampling steps', padL + 62, 14);

            // total-variation-ish error against the target
            var err = 0;
            for (i = 0; i < NB; i++) {
                var xc = X0 + (i + 0.5) * (X1 - X0) / NB;
                err += Math.abs(hist[i] / dens - targetPdf(xc)) * (X1 - X0) / NB;
            }
            r1.textContent = 'S = ' + S + ' of ' + DDPM_T + ' steps  (' + (DDPM_T / S).toFixed(0) +
                '× fewer network calls)     mismatch with the target: ' + (err / 2).toFixed(4);
            r2.textContent = 'same starting point, 8 runs → spread ' + psd.toExponential(2) +
                (stochastic ? '   (noise injected each step)' : '   (identical every time)');

            if (!stochastic && psd < 1e-12) {
                verdict.textContent = '✓ Deterministic: one x_T gives exactly one x₀, every time. That is what makes inversion and image editing possible.';
                verdict.style.color = '#0e7490';
            } else if (stochastic) {
                verdict.textContent = '✗ Stochastic (this is DDPM): the same x_T lands somewhere different each run, so there is no map to invert.';
                verdict.style.color = '#c0392b';
            } else {
                verdict.textContent = 'Deterministic to ' + psd.toExponential(1) + ' — floating point, not noise.';
                verdict.style.color = '#0e7490';
            }
        }

        bMode.addEventListener('click', function () {
            stochastic = !stochastic;
            bMode.textContent = stochastic ? 'σ = DDPM  (stochastic)' : 'σ = 0  (DDIM, deterministic)';
            bMode.className = stochastic ? 'secondary' : '';
            draw();
        });
        bRun.addEventListener('click', function () { reseed(); draw(); });
        sS.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 34 (Lecture 11): KDE and the bandwidth trade-off
    // ─────────────────────────────────────────────────────────
    function initKde() {
        var root = byId('widget-kde');
        if (!root) return;
        makeTitle(root, 'Try it — the bandwidth is the whole model');

        var DATA = [2.0, 2.1, 2.2, 8.0, 8.1, 8.2];
        var sH = slider(root, 'bandwidth h', 0.03, 4, 0.01, 0.2);

        var canvas = makeCanvas(root, 230);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Six data points in two tight clusters, and a Gaussian dropped on each one. Kernel density estimation is ' +
            'nothing more than adding those bumps up and dividing by how many there are — no training, no parameters ' +
            'beyond the single width h. Slide h down and the estimate becomes six isolated spikes: it has memorised the ' +
            'training set and assigns essentially zero density to anything between the points, which is overfitting in ' +
            'its purest visible form. Slide h up and the bumps merge, and something worse happens — watch the density at ' +
            'x = 5, exactly where no data exists. Past about h = 3 the estimator assigns *more* density to that empty gap ' +
            'than to the clusters where all six points actually live. That is the failure mode to remember: over-smoothing ' +
            'does not merely blur the answer, it invents mass in regions the data explicitly avoided.');

        function kde(x, h) {
            var s2 = 0;
            for (var i = 0; i < DATA.length; i++) {
                var u = (x - DATA[i]) / h;
                s2 += Math.exp(-0.5 * u * u) / (h * Math.sqrt(2 * Math.PI));
            }
            return s2 / DATA.length;
        }

        function draw() {
            var h = +sH.value;
            sH._val.textContent = h.toFixed(2);
            var w = canvas._w, hh = canvas._h;
            var padL = 30, padR = 16, baseY = hh - 36, topY = 22;
            var X0 = -1, X1 = 11;
            function px(x) { return padL + (x - X0) / (X1 - X0) * (w - padL - padR); }
            var maxV = 0.6;
            function py(v) { return baseY - Math.min(v / maxV, 1) * (baseY - topY); }

            ctx.clearRect(0, 0, w, hh);
            // individual kernels
            ctx.lineWidth = 1;
            DATA.forEach(function (xi) {
                ctx.beginPath();
                for (var x = X0; x <= X1; x += 0.02) {
                    var u = (x - xi) / h;
                    var v = Math.exp(-0.5 * u * u) / (h * Math.sqrt(2 * Math.PI)) / DATA.length;
                    var y = py(v);
                    x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
                }
                ctx.strokeStyle = 'rgba(120,120,120,0.35)'; ctx.stroke();
            });
            // the sum
            ctx.beginPath();
            for (var x = X0; x <= X1; x += 0.02) {
                var y = py(kde(x, h));
                x === X0 ? ctx.moveTo(px(x), y) : ctx.lineTo(px(x), y);
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.4; ctx.stroke();

            ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(w - padR, baseY); ctx.stroke();
            DATA.forEach(function (xi) {
                ctx.strokeStyle = '#111'; ctx.lineWidth = 1.6;
                ctx.beginPath(); ctx.moveTo(px(xi), baseY); ctx.lineTo(px(xi), baseY + 8); ctx.stroke();
            });
            // the empty middle
            var mid = kde(5, h), peak = kde(2.1, h);
            ctx.strokeStyle = 'rgba(192,57,43,0.6)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.3;
            ctx.beginPath(); ctx.moveTo(px(5), topY); ctx.lineTo(px(5), baseY); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#c0392b'; ctx.font = '9.6px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('x = 5, no data here', px(5), topY - 6);
            ctx.fillStyle = '#999';
            ctx.fillText('the six samples', px(5), baseY + 28);

            r1.textContent = 'density at a cluster (x=2.1): ' + peak.toFixed(4) +
                '     density in the empty gap (x=5): ' + mid.toFixed(4) +
                '     ratio: ' + (mid / peak).toFixed(3);

            if (h < 0.12) {
                verdict.textContent = '✗ Under-smoothed: six spikes on six points, near-zero everywhere else. It has memorised the sample, not learned a density.';
                verdict.style.color = '#c0392b';
            } else if (mid > peak) {
                verdict.textContent = '✗ Over-smoothed past the point of absurdity: the empty gap at x = 5 now has HIGHER density than the clusters.';
                verdict.style.color = '#c0392b';
            } else if (mid / peak > 0.3) {
                verdict.textContent = '⚠ The two clusters have merged — substantial mass is being assigned to a region the data avoided.';
                verdict.style.color = '#d97706';
            } else {
                verdict.textContent = '✓ Two modes resolved, gap correctly near-empty. This is roughly the useful range of h.';
                verdict.style.color = '#0e7490';
            }
        }
        sH.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 35 (Lecture 11): change of variables in 1-D
    // ─────────────────────────────────────────────────────────
    function initChangeVars() {
        var root = byId('widget-changevars');
        if (!root) return;
        makeTitle(root, 'Try it — stretch the axis and watch the density thin out');

        var maps = [
            { name: 'x = a·z + b', f: function (z, a, b) { return a * z + b; },
              finv: function (x, a, b) { return (x - b) / a; }, dfinv: function (x, a, b) { return 1 / a; } },
            { name: 'x = z³ + b', f: function (z, a, b) { return a * z * z * z + b; },
              finv: function (x, a, b) { return Math.cbrt((x - b) / a); },
              dfinv: function (x, a, b) { var c = Math.cbrt((x - b) / a); return 1 / (3 * a * c * c); } },
            { name: 'x = a·sinh(z) + b', f: function (z, a, b) { return a * Math.sinh(z) + b; },
              finv: function (x, a, b) { return Math.asinh((x - b) / a); },
              dfinv: function (x, a, b) { var u = (x - b) / a; return 1 / (a * Math.sqrt(u * u + 1)); } }
        ];
        var cur = 0;

        var controls = make('div', null, root);
        var btns = maps.map(function (m, i) {
            var b = make('button', i === 0 ? null : 'secondary', controls, m.name);
            b.addEventListener('click', function () {
                cur = i;
                btns.forEach(function (bb, j) { bb.className = j === i ? '' : 'secondary'; });
                draw();
            });
            return b;
        });
        var sA = slider(root, 'a', -3, 3, 0.05, 1.6);
        var sB = slider(root, 'b', -3, 3, 0.05, 0.5);

        var canvas = makeCanvas(root, 260);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Left: the base density, a standard normal, which we can sample and evaluate trivially. Right: what it ' +
            'becomes after the map, computed two ways — the teal curve is the change-of-variables formula and the orange ' +
            'histogram is ten thousand actual samples pushed through. They agree, which is the formula confirmed rather ' +
            'than asserted. The thing to watch is the correction factor. Where the map stretches the axis, the same ' +
            'probability mass has to cover more ground, so the density drops; where it compresses, the density spikes. ' +
            'That is the entire content of the Jacobian term. Now drag a below zero: the map flips left and right, the ' +
            'derivative goes negative, and yet the density stays positive everywhere — which is exactly why the formula ' +
            'takes an absolute value. Without it you would be claiming a negative probability density, which is not a ' +
            'thing. The cubic is worth a look too: its derivative vanishes at the origin, so the correction factor blows ' +
            'up and the density has an integrable spike there.');

        function draw() {
            var a = +sA.value, b = +sB.value, m = maps[cur];
            if (Math.abs(a) < 0.08) a = a < 0 ? -0.08 : 0.08;
            sA._val.textContent = a.toFixed(2);
            sB._val.textContent = b.toFixed(2);

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var gap = 34, colW = (w - gap - 30) / 2;
            var baseY = h - 40, topY = 26;

            // left: base density
            var Z0 = -3.6, Z1 = 3.6, maxP = 0.45;
            function pxL(z) { return 20 + (z - Z0) / (Z1 - Z0) * colW; }
            function pyL(v) { return baseY - Math.min(v / maxP, 1) * (baseY - topY); }
            ctx.beginPath();
            for (var z = Z0; z <= Z1; z += 0.02) {
                var y = pyL(gaussPdf(z, 0, 1));
                z === Z0 ? ctx.moveTo(pxL(z), y) : ctx.lineTo(pxL(z), y);
            }
            ctx.strokeStyle = '#999'; ctx.lineWidth = 2.2; ctx.stroke();
            ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(20, baseY); ctx.lineTo(20 + colW, baseY); ctx.stroke();
            ctx.font = '10.4px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = '#777';
            ctx.fillText('base  p(z) = 𝒩(0,1)', 20 + colW / 2, 16);

            // right: transformed
            var X0 = -8, X1 = 8, maxQ = 0.6;
            var x0 = 20 + colW + gap;
            function pxR(x) { return x0 + (x - X0) / (X1 - X0) * colW; }
            function pyR(v) { return baseY - Math.min(v / maxQ, 1) * (baseY - topY); }

            // histogram of pushed-forward samples
            var NB = 70, hist = new Array(NB), i, N = 10000;
            for (i = 0; i < NB; i++) hist[i] = 0;
            for (i = 0; i < N; i++) {
                var zz = nrand(), xx = m.f(zz, a, b);
                var bi = Math.floor((xx - X0) / (X1 - X0) * NB);
                if (bi >= 0 && bi < NB) hist[bi]++;
            }
            var bw = (X1 - X0) / NB, cw = colW / NB;
            for (i = 0; i < NB; i++) {
                var dens = hist[i] / (N * bw);
                var hh2 = Math.min(dens / maxQ, 1) * (baseY - topY);
                ctx.fillStyle = 'rgba(217,119,6,0.45)';
                ctx.fillRect(x0 + i * cw, baseY - hh2, cw, hh2);
            }
            // formula
            ctx.beginPath();
            var started = false;
            for (var x = X0; x <= X1; x += 0.01) {
                var zi = m.finv(x, a, b), d = Math.abs(m.dfinv(x, a, b));
                if (!isFinite(zi) || !isFinite(d)) { started = false; continue; }
                var q = gaussPdf(zi, 0, 1) * d;
                if (!isFinite(q)) { started = false; continue; }
                var yy = pyR(q);
                started ? ctx.lineTo(pxR(x), yy) : ctx.moveTo(pxR(x), yy);
                started = true;
            }
            ctx.strokeStyle = TEAL; ctx.lineWidth = 2.4; ctx.stroke();
            ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x0, baseY); ctx.lineTo(x0 + colW, baseY); ctx.stroke();
            ctx.font = '10.4px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillStyle = TEAL; ctx.fillText('formula q(x)', x0 + colW / 2 - 46, 16);
            ctx.fillStyle = ORANGE; ctx.fillText('samples', x0 + colW / 2 + 44, 16);

            // the correction factor at a probe point
            var probe = b + 0.6 * (a >= 0 ? 1 : -1);
            var corr = Math.abs(m.dfinv(probe, a, b));
            r1.textContent = 'at x = ' + probe.toFixed(2) + ':  |dz/dx| = ' + corr.toFixed(4) +
                '   →  density ' + (corr > 1 ? 'multiplied up' : 'thinned') + ' by that factor';

            if (a < 0) {
                verdict.textContent = '✓ a < 0 flips the axis, so dx/dz is negative — and the density is still positive everywhere. That is what the absolute value is for.';
                verdict.style.color = '#0e7490';
            } else if (cur === 1) {
                verdict.textContent = '✓ The cubic flattens at z = 0, so |dz/dx| → ∞ there and the density spikes — integrable, but unbounded.';
                verdict.style.color = '#0e7490';
            } else {
                verdict.textContent = '✓ Formula and samples agree. Stretch the axis (|a| > 1) and the density drops; compress it and the density rises.';
                verdict.style.color = '#0e7490';
            }
        }
        [sA, sB].forEach(function (x) { x.addEventListener('input', draw); });
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget 36 (Lecture 11): affine coupling layers, stacked
    // ─────────────────────────────────────────────────────────
    function initCoupling() {
        var root = byId('widget-coupling');
        if (!root) return;
        makeTitle(root, 'Try it — stack coupling layers and watch a Gaussian become a target');

        var sL = slider(root, 'coupling layers', 0, 8, 1, 0);
        var canvas = makeCanvas(root, 280);
        var ctx = canvas.getContext('2d');
        var r1 = make('div', 'w-readout', root, '');
        var verdict = make('div', 'verdict', root, '');

        make('div', 'w-note', root,
            'Each coupling layer splits the two coordinates, leaves one completely untouched, and rescales and shifts ' +
            'the other by amounts computed from the one it left alone. That asymmetry is the whole trick. Because the ' +
            'first coordinate passes through unchanged, the Jacobian is triangular and its determinant is just the ' +
            'product of the scale factors — no matter how complicated the network computing them is. And because the ' +
            'scale and shift depend only on the untouched half, the layer inverts exactly by running the same network ' +
            'again and undoing the arithmetic, so the network itself never has to be invertible. One layer alone can ' +
            'only ever move one axis, which is why the picture barely changes at first; alternate which half is frozen ' +
            'and stack a few, and the Gaussian bends into the ring. The running log-determinant printed below is exactly ' +
            'the term that would be added to the log-likelihood.');

        var N = 1200, seeds = [];
        (function () {
            var sd = 987654321;
            function r() { sd = (sd * 1103515245 + 12345) & 0x7fffffff; return sd / 0x7fffffff; }
            function rn() { return Math.sqrt(-2 * Math.log(r() || 1e-9)) * Math.cos(2 * Math.PI * r()); }
            for (var i = 0; i < N; i++) seeds.push([rn(), rn()]);
        })();

        // hand-set coupling parameters that bend a Gaussian toward a ring
        function layer(p, k) {
            var u = p[0], v = p[1];
            var swap = k % 2 === 1;
            var a = swap ? v : u, bcoord = swap ? u : v;
            var s = 0.45 * Math.sin(1.1 * a + 0.6 * k) - 0.12;
            var t = 1.05 * Math.sin(0.9 * a + 0.4 * k) + 0.25 * a;
            var nb = bcoord * Math.exp(s) + t;
            return { p: swap ? [nb, v] : [u, nb], s: s };
        }

        function draw() {
            var L = +sL.value;
            sL._val.textContent = L;
            var pts = [], logdet = 0;
            for (var i = 0; i < N; i++) {
                var p = [seeds[i][0], seeds[i][1]], ld = 0;
                for (var k = 0; k < L; k++) { var out = layer(p, k); p = out.p; ld += out.s; }
                pts.push(p); logdet += ld;
            }
            logdet /= N;

            var w = canvas._w, h = canvas._h;
            ctx.clearRect(0, 0, w, h);
            var cx = w / 2, cy = h / 2, sc = Math.min(w, h) / 11;
            function px(x) { return cx + x * sc; }
            function py(y) { return cy - y * sc; }
            ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(px(-5), py(0)); ctx.lineTo(px(5), py(0)); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(px(0), py(-4)); ctx.lineTo(px(0), py(4)); ctx.stroke();

            pts.forEach(function (p) {
                ctx.fillStyle = 'rgba(14,116,144,0.42)';
                ctx.beginPath(); ctx.arc(px(p[0]), py(p[1]), 1.9, 0, 7); ctx.fill();
            });

            ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillStyle = '#777';
            ctx.fillText(L === 0 ? 'the base: 𝒩(0, I)' : L + ' coupling layer' + (L === 1 ? '' : 's') + ' applied', 12, 16);

            r1.textContent = 'layers: ' + L + '     mean Σ log-scale (the log-determinant term): ' + logdet.toFixed(4);
            if (L === 0) {
                verdict.textContent = 'Just the base distribution. Every layer from here is invertible in closed form and has a triangular Jacobian.';
                verdict.style.color = '#999';
            } else if (L < 3) {
                verdict.textContent = 'One layer moves one coordinate only — that is why so little has happened. Expressiveness comes from stacking with the roles swapped.';
                verdict.style.color = '#d97706';
            } else {
                verdict.textContent = '✓ The Gaussian has been bent into something a single affine map could never produce, and every step is still exactly invertible.';
                verdict.style.color = '#0e7490';
            }
        }
        sL.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    function init() {
        initKde();
        initChangeVars();
        initCoupling();
        initTimeEmbed();
        initGuidance();
        initDdim();
        initForward();
        initDdpmPost();
        initReverse();
        initPriorCover();
        initVq();
        initReparam();
        initVaeLatent();
        initJsVsW();
        initW1();
        initElbo();
        initEm();
        initToy();
        initFixedClf();
        initModeCollapse();
        initCgan();
        initCycle();
        initTransConv();
        initExperiment();
        initSigma();
        initPdfCdf();
        initFit();
        initKl();
        initWarp();
        initConjugate();
        initFdiv();
        initJensen();
        initGanConj();
        initOptD();
        initGanTrain();
        initUat();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
