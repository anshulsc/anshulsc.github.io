/**
 * CS294-158 Notes — interactive widgets.
 *
 * Each widget mounts into an empty <div class="note-widget" id="widget-…"> that
 * lives inside the rendered markdown, and returns silently when its div is
 * absent — so this one file serves every chapter in the series.
 *
 * Helper functions are the generic ones from deep-gen/dgm-widgets.js. The
 * widgets themselves are written for this course: their labels use this
 * chapter's notation table, not another series'.
 */

(function () {
    'use strict';

    var BLUE = '#1a4f7a';        // real / true things
    var BLUE_FILL = 'rgba(26,79,122,0.14)';
    var GOLD = '#b5761a';        // model / generated things
    var GOLD_FILL = 'rgba(181,118,26,0.16)';
    var GREY = '#8a8a8a';

    function byId(id) { return document.getElementById(id); }

    function make(tag, className, parent, text) {
        var e = document.createElement(tag);
        if (className) e.className = className;
        if (text !== undefined) e.textContent = text;
        if (parent) parent.appendChild(e);
        return e;
    }

    function makeTitle(root, text) { make('div', 'w-title', root, text); }

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
        make('label', null, wrap, labelText + ' ');
        var input = make('input', null, wrap);
        input.type = 'range';
        input.min = min; input.max = max; input.step = step; input.value = value;
        var val = make('span', 'w-slider-val', wrap, value);
        input._val = val;
        return input;
    }

    var log2 = function (x) { return Math.log(x) / Math.LN2; };

    // ─────────────────────────────────────────────────────────
    // Widget (Lecture 1): the cost of being wrong, in bits
    //
    // A biased coin with true P(heads) = p, coded with a model that believes q.
    // Shows that the average cost is H(p) + D_KL(p ‖ q), bottoming out at H(p).
    // ─────────────────────────────────────────────────────────
    function initBits() {
        var root = byId('widget-bits');
        if (!root) return;

        makeTitle(root, 'Coding a biased coin with the wrong model');
        var canvas = makeCanvas(root, 260);
        var ctx = canvas.getContext('2d');
        var sp = slider(root, 'true p (heads)', 0.02, 0.98, 0.01, 0.9);
        var sq = slider(root, 'model q (heads)', 0.02, 0.98, 0.01, 0.5);
        var note = make('div', 'w-note', root);

        function H(p) { return -(p * log2(p) + (1 - p) * log2(1 - p)); }
        function KL(p, q) { return p * log2(p / q) + (1 - p) * log2((1 - p) / (1 - q)); }

        function draw() {
            var p = parseFloat(sp.value), q = parseFloat(sq.value);
            sp._val.textContent = p.toFixed(2);
            sq._val.textContent = q.toFixed(2);

            var h = H(p), kl = KL(p, q), ce = h + kl;
            var w = canvas._w, hgt = canvas._h;
            ctx.clearRect(0, 0, w, hgt);

            // --- the bar: entropy floor + KL excess ---
            var x0 = 60, barTop = 34, barH = 42;
            var scale = (w - x0 - 30) / 2.2;            // 2.2 bits full width
            ctx.fillStyle = BLUE_FILL;
            ctx.fillRect(x0, barTop, h * scale, barH);
            ctx.strokeStyle = BLUE; ctx.lineWidth = 1.4;
            ctx.strokeRect(x0, barTop, h * scale, barH);
            ctx.fillStyle = GOLD_FILL;
            ctx.fillRect(x0 + h * scale, barTop, kl * scale, barH);
            ctx.strokeStyle = GOLD;
            ctx.strokeRect(x0 + h * scale, barTop, kl * scale, barH);

            ctx.font = '11px Inter, sans-serif';
            ctx.fillStyle = BLUE; ctx.textAlign = 'left';
            ctx.fillText('H(p) = ' + h.toFixed(3), x0 + 6, barTop - 8);
            ctx.fillStyle = GOLD;
            ctx.fillText('D_KL = ' + kl.toFixed(3), x0 + h * scale + 6, barTop + barH + 15);
            ctx.fillStyle = '#333'; ctx.font = '12px Inter, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText('cost', x0 - 10, barTop + 26);

            // --- the curve: cost against q, with the minimum marked ---
            var gy = 130, gh = 96, gx = x0, gw = w - x0 - 30;
            ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(gx, gy + gh); ctx.lineTo(gx + gw, gy + gh); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(gx, gy + gh); ctx.stroke();

            // The curve runs off the top for a badly wrong q. Clamping would draw a
            // flat cap that reads as part of the function, so the path breaks instead.
            var maxC = 2.2;
            ctx.strokeStyle = GOLD; ctx.lineWidth = 1.8; ctx.beginPath();
            var drawing = false;
            for (var i = 0; i <= 200; i++) {
                var qq = 0.02 + (0.96 * i) / 200;
                var c = H(p) + KL(p, qq);
                if (c > maxC) { drawing = false; continue; }
                var px = gx + (qq * gw), py = gy + gh - (c / maxC) * gh;
                if (!drawing) { ctx.moveTo(px, py); drawing = true; } else ctx.lineTo(px, py);
            }
            ctx.stroke();

            // entropy floor
            var fy = gy + gh - (h / maxC) * gh;
            ctx.strokeStyle = BLUE; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.2;
            ctx.beginPath(); ctx.moveTo(gx, fy); ctx.lineTo(gx + gw, fy); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = BLUE; ctx.font = '10.5px Inter, sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('floor H(p) — no code beats this', gx + 6, fy - 5);

            // current q
            var cx = gx + q * gw, cy = gy + gh - (Math.min(ce, maxC) / maxC) * gh;
            ctx.fillStyle = GOLD;
            ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 6.2832); ctx.fill();
            ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, gy + gh); ctx.stroke();

            ctx.fillStyle = GREY; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('model q', gx + gw / 2, gy + gh + 16);

            note.innerHTML = 'Coding flips from a coin with p = <strong>' + p.toFixed(2) +
                '</strong> using a code built for q = <strong>' + q.toFixed(2) + '</strong> costs <strong>' +
                ce.toFixed(3) + ' bits per flip</strong> — the unavoidable H(p) = ' + h.toFixed(3) +
                ' plus D_KL = ' + kl.toFixed(3) + ' paid purely for being wrong. Drag q onto p and the gold ' +
                'section vanishes: the curve touches the floor exactly once, at q = p, which is why minimising ' +
                'the file size and fitting the distribution are the same optimisation.';
        }

        sp.addEventListener('input', draw);
        sq.addEventListener('input', draw);
        draw();
    }

    // ─────────────────────────────────────────────────────────
    // Widget (Lecture 1): why MAE has to mask 75%
    //
    // Masks a structured grid, then reconstructs each hidden cell by averaging
    // its visible neighbours. At low ratios that trivially works, so the task
    // teaches nothing; the error only becomes large once neighbours are gone.
    // ─────────────────────────────────────────────────────────
    function initMask() {
        var root = byId('widget-mask');
        if (!root) return;

        makeTitle(root, 'Interpolation solves a lightly-masked image, and only that');
        var canvas = makeCanvas(root, 344);
        var ctx = canvas.getContext('2d');
        var sr = slider(root, 'masked fraction', 0.05, 0.95, 0.05, 0.25);
        var note = make('div', 'w-note', root);

        var N = 24, scene = [];
        (function buildScene() {           // smooth blobs — stands in for pixel redundancy
            for (var y = 0; y < N; y++) {
                scene[y] = [];
                for (var x = 0; x < N; x++) {
                    // Low frequencies on purpose: the whole argument is that pixels
                    // are redundant, so the scene has to actually be smooth.
                    var v = 0.5 + 0.30 * Math.sin(x / 7.5) * Math.cos(y / 6.2)
                          + 0.16 * Math.sin((x + y) / 11.0)
                          + 0.10 * Math.cos((x - y) / 9.0);
                    scene[y][x] = Math.max(0, Math.min(1, v));
                }
            }
        })();

        // A fixed pseudo-random order, so sliding the ratio reveals/hides cells
        // stably instead of reshuffling the whole mask on every frame.
        var order = [];
        (function () {
            for (var i = 0; i < N * N; i++) order.push(i);
            var seed = 7;
            for (var j = order.length - 1; j > 0; j--) {
                seed = (seed * 1103515245 + 12345) % 2147483648;
                var k = seed % (j + 1);
                var t = order[j]; order[j] = order[k]; order[k] = t;
            }
        })();

        function shade(v) {
            var g = Math.round(255 - v * 150);
            return 'rgb(' + Math.round(g * 0.82) + ',' + Math.round(g * 0.88) + ',' + g + ')';
        }

        function draw() {
            var ratio = parseFloat(sr.value);
            sr._val.textContent = Math.round(ratio * 100) + '%';

            var hidden = {};
            var nHide = Math.round(ratio * N * N);
            for (var i = 0; i < nHide; i++) hidden[order[i]] = true;

            var w = canvas._w;
            var cell = Math.floor(Math.min((w - 40) / (2 * N + 4), 9));
            var gw = cell * N, ox = 14, oy = 34, ox2 = ox + gw + 26;

            ctx.clearRect(0, 0, w, canvas._h);
            ctx.font = '11px Inter, sans-serif'; ctx.textAlign = 'left'; ctx.fillStyle = '#555';
            ctx.fillText('masked input', ox, oy - 10);
            ctx.fillText('neighbour-average reconstruction', ox2, oy - 10);

            var err = 0, count = 0;
            for (var y = 0; y < N; y++) {
                for (var x = 0; x < N; x++) {
                    var idx = y * N + x, isHidden = hidden[idx];
                    ctx.fillStyle = isHidden ? '#efefef' : shade(scene[y][x]);
                    ctx.fillRect(ox + x * cell, oy + y * cell, cell - 1, cell - 1);

                    var val;
                    if (!isHidden) {
                        val = scene[y][x];
                    } else {                       // average the visible 8-neighbourhood
                        var sum = 0, n = 0;
                        for (var dy = -1; dy <= 1; dy++) {
                            for (var dx = -1; dx <= 1; dx++) {
                                var yy = y + dy, xx = x + dx;
                                if (yy < 0 || xx < 0 || yy >= N || xx >= N) continue;
                                if (hidden[yy * N + xx]) continue;
                                sum += scene[yy][xx]; n++;
                            }
                        }
                        val = n ? sum / n : 0.5;    // nothing visible nearby: no better than a guess
                        err += Math.abs(val - scene[y][x]); count++;
                    }
                    ctx.fillStyle = shade(val);
                    ctx.fillRect(ox2 + x * cell, oy + y * cell, cell - 1, cell - 1);
                }
            }

            var mae = count ? err / count : 0;
            var by = oy + gw + 26, bw = w - 28;
            ctx.fillStyle = '#eee'; ctx.fillRect(ox, by, bw, 10);
            ctx.fillStyle = mae > 0.09 ? BLUE : GOLD;
            ctx.fillRect(ox, by, Math.min(1, mae / 0.22) * bw, 10);
            ctx.fillStyle = '#555'; ctx.font = '11px Inter, sans-serif';
            ctx.fillText('mean error of the interpolation: ' + mae.toFixed(3), ox, by + 26);

            note.innerHTML = 'At <strong>' + Math.round(ratio * 100) + '%</strong> masked, averaging each hidden ' +
                'cell\'s visible neighbours reconstructs it to within <strong>' + mae.toFixed(3) + '</strong> on ' +
                'average. Below about 40% the copy is nearly perfect — a network trained on that objective can ' +
                'score well as a smoothing filter and never learn what the image contains. Push past 75% and the ' +
                'neighbours are gone, the shortcut collapses, and the only way to fill the gaps is to recognise ' +
                'the structure. The ratio is a dial on how much information the answer actually requires.';
        }

        sr.addEventListener('input', draw);
        draw();
    }

    function init() {
        initBits();
        initMask();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
