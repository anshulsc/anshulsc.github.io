/**
 * The Loop — a small Möbius strip ornament ("it's inverted!").
 *
 * Pure canvas 2D with a hand-rolled 3D projection, no dependencies.
 * A smooth holographic Möbius ribbon — the surface is rendered as
 * depth-sorted, normal-shaded slices so it reads as one continuous
 * curved sheet, with particles flowing along it as glints.
 *
 *  - Accent color follows the section being read.
 *  - Rotation tracks scroll progress through the page.
 *  - Hover speeds the flow up.
 *  - Click: the strip untwists and unrolls into a flat ribbon while the
 *    page returns to the top, then winds itself back into the loop.
 */
(function () {
    'use strict';

    var mount = document.getElementById('visual-canvas');
    if (!mount) return;

    var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    mount.appendChild(canvas);

    // --- Sizing (DPR aware) ---
    var size = 0, dpr = 1, R = 0;
    function resize() {
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        size = mount.clientWidth;
        canvas.width = size * dpr;
        canvas.height = size * dpr;
        R = canvas.width / 2;
    }

    // --- Section accents (same palette as the page) ---
    var SECTION_COLORS = [
        [110, 70, 190],  // intro       #6e46be
        [23, 114, 208],  // experience  #1772d0
        [240, 146, 40],  // research    #f09228
        [70, 70, 70]     // reflections
    ];
    var accent = SECTION_COLORS[0].slice();
    var targetAccent = SECTION_COLORS[0];
    var scrollFrac = 0, scrollRot = 0;

    window.addEventListener('scroll', function () {
        var doc = document.documentElement;
        var max = Math.max(doc.scrollHeight - window.innerHeight, 1);
        scrollFrac = Math.min(1, Math.max(0, window.scrollY / max));
        var idx = Math.min(3, Math.floor(scrollFrac * 4));
        targetAccent = SECTION_COLORS[idx];
    }, { passive: true });

    // --- Geometry (normalized units, projected onto R) ---
    var TAU = Math.PI * 2;    // the surface closes after 2π (with a flip)
    var RM = 0.54;            // ring radius
    var W = 0.26;             // strip half-width
    var F = 3.4;              // perspective distance
    var TILT = -0.52;         // view tilt (rotX)
    var ROLL = -0.38;         // canvas roll for character
    var K = 0.88;             // fit margin

    // Möbius surface point → world [x, y, z]; tw = twist (1 = Möbius, 0 = flat band)
    function surf(u, v, tw) {
        var half = (u / 2) * tw;
        var rr = RM + v * W * Math.cos(half);
        return [
            rr * Math.cos(u),
            v * W * Math.sin(half),
            rr * Math.sin(u)
        ];
    }

    // world → camera space [x, y, z] (z toward viewer)
    function toCam(pt, rotY) {
        var cA = Math.cos(rotY), sA = Math.sin(rotY);
        var x = pt[0] * cA + pt[2] * sA;
        var z = -pt[0] * sA + pt[2] * cA;
        var cB = Math.cos(TILT), sB = Math.sin(TILT);
        var y = pt[1] * cB - z * sB;
        z = pt[1] * sB + z * cB;
        return [x, y, z];
    }

    // camera space → screen {x, y, d}, d = 0(far)..1(near)
    function project(c) {
        var s = F / (F - c[2]);
        return {
            x: c[0] * s * K * R,
            y: -c[1] * s * K * R,
            d: Math.min(1, Math.max(0, (c[2] + RM + W) / (2 * (RM + W))))
        };
    }

    // flat-ribbon screen position (the unrolled state)
    function flat(u, v) {
        var uu = ((u % TAU) + TAU) % TAU;
        if (uu === 0 && u > 0) uu = TAU; // keep the seam sample at the far end
        return {
            x: (uu / TAU - 0.5) * 1.9 * K * R,
            y: v * W * 1.35 * K * R,
            d: 0.72
        };
    }

    function lerpPt(a, b, e) {
        return { x: a.x + (b.x - a.x) * e, y: a.y + (b.y - a.y) * e, d: a.d + (b.d - a.d) * e };
    }

    function pos(u, v, rotY, e, twist) {
        var m = project(toCam(surf(u, v, twist), rotY));
        return e > 0 ? lerpPt(m, flat(u, v), e) : m;
    }

    // --- Particles flowing along the surface ---
    var COUNT = 90;
    var particles = [];
    for (var i = 0; i < COUNT; i++) {
        particles.push({
            u: Math.random() * TAU,
            v: (Math.random() * 2 - 1) * 0.85,
            sz: 0.6 + Math.random() * 0.9,
            sp: 0.8 + Math.random() * 0.5,
            tw: Math.random() * Math.PI * 2
        });
    }

    // --- Interaction ---
    var speedBoost = 1, targetBoost = 1;
    mount.addEventListener('mouseenter', function () { targetBoost = 1.9; });
    mount.addEventListener('mouseleave', function () { targetBoost = 1; });

    // unwrap: 0 = Möbius, 1 = flat ribbon
    var unwrap = 0, unwrapTarget = 0, holdT = 0;

    function unwind() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
        if (reduceMotion) return;
        unwrapTarget = 1;
        holdT = 0;
    }
    mount.addEventListener('click', unwind);
    mount.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); unwind(); }
    });

    function ease(p) {
        return p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
    }

    // --- Rendering ---
    var SLICES = 72;

    function render(t) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.translate(R, R);
        ctx.rotate(ROLL);

        var e = ease(unwrap);
        var twist = 1 - e;
        var rotY = t * 0.32 + scrollRot;
        var wob = Math.sin(t * 0.4) * 0.1; // gentle breathing tilt
        rotY += wob;

        var dark = [(accent[0] * 0.55) | 0, (accent[1] * 0.55) | 0, (accent[2] * 0.55) | 0];

        // soft ambient glow
        var glow = ctx.createRadialGradient(0, 0, R * 0.1, 0, 0, R);
        glow.addColorStop(0, 'rgba(' + accent[0] + ',' + accent[1] + ',' + accent[2] + ',0.07)');
        glow.addColorStop(1, 'rgba(' + accent[0] + ',' + accent[1] + ',' + accent[2] + ',0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(0, 0, R, 0, Math.PI * 2);
        ctx.fill();

        // --- the surface: depth-sorted shaded slices ---
        var slices = [];
        var j, u0, u1;
        for (j = 0; j < SLICES; j++) {
            u0 = (j / SLICES) * TAU;
            u1 = ((j + 1) / SLICES) * TAU;

            // camera-space corners for shading
            var c00 = toCam(surf(u0, -1, twist), rotY);
            var c01 = toCam(surf(u0, 1, twist), rotY);
            var c10 = toCam(surf(u1, -1, twist), rotY);

            // facing: |z of normal| (two-sided sheet)
            var ax = c10[0] - c00[0], ay = c10[1] - c00[1], az = c10[2] - c00[2];
            var bx = c01[0] - c00[0], by = c01[1] - c00[1], bz = c01[2] - c00[2];
            var nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
            var nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
            var shade = Math.abs(nz) / nl;

            slices.push({
                u0: u0, u1: u1,
                z: (c00[2] + c01[2]) / 2,
                shade: shade
            });
        }
        slices.sort(function (a, b) { return a.z - b.z; }); // far → near

        var overlap = TAU / SLICES * 0.15; // hide seams between slices
        for (j = 0; j < SLICES; j++) {
            var sl = slices[j];
            var p00 = pos(sl.u0 - overlap, -1, rotY, e, twist);
            var p01 = pos(sl.u0 - overlap, 1, rotY, e, twist);
            var p11 = pos(sl.u1 + overlap, 1, rotY, e, twist);
            var p10 = pos(sl.u1 + overlap, -1, rotY, e, twist);

            var d = (p00.d + p11.d) / 2;
            // translucent hologram sheet: nearer and more face-on = deeper tint
            var a = (0.05 + 0.30 * d) * (0.35 + 0.65 * sl.shade);

            ctx.fillStyle = 'rgba(' + accent[0] + ',' + accent[1] + ',' + accent[2] + ',' + a.toFixed(3) + ')';
            ctx.beginPath();
            ctx.moveTo(p00.x, p00.y);
            ctx.lineTo(p01.x, p01.y);
            ctx.lineTo(p11.x, p11.y);
            ctx.lineTo(p10.x, p10.y);
            ctx.closePath();
            ctx.fill();
        }

        // --- the boundary edges: crisp continuous contours ---
        ctx.lineJoin = 'round';
        ctx.lineWidth = 1.4 * dpr;
        var side, sgn;
        for (side = 0; side < 2; side++) {
            sgn = side === 0 ? 1 : -1;
            var M = 110, prev = null;
            for (j = 0; j <= M; j++) {
                var u = (j / M) * TAU;
                var p = pos(u, sgn, rotY, e, twist);
                if (prev) {
                    var al = 0.18 + 0.55 * ((p.d + prev.d) / 2);
                    ctx.strokeStyle = 'rgba(' + dark[0] + ',' + dark[1] + ',' + dark[2] + ',' + al.toFixed(3) + ')';
                    ctx.beginPath();
                    ctx.moveTo(prev.x, prev.y);
                    ctx.lineTo(p.x, p.y);
                    ctx.stroke();
                }
                prev = p;
            }
        }

        // --- particles: glints riding the flow ---
        for (i = 0; i < COUNT; i++) {
            var pt = particles[i];
            var q = pos(pt.u, pt.v, rotY, e, twist);
            var twk = 0.85 + 0.15 * Math.sin(t * 3 + pt.tw);

            var wgt = q.d * q.d * 0.75;
            var cr = accent[0] * (1 - wgt * 0.8);
            var cg = accent[1] * (1 - wgt * 0.8);
            var cb = accent[2] * (1 - wgt * 0.8);
            var pa = (0.20 + 0.60 * q.d) * twk;

            ctx.fillStyle = 'rgba(' + (cr | 0) + ',' + (cg | 0) + ',' + (cb | 0) + ',' + pa.toFixed(3) + ')';
            ctx.beginPath();
            ctx.arc(q.x, q.y, pt.sz * dpr * (0.5 + 0.7 * q.d), 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.restore();
    }

    // --- Simulation ---
    function step(dt, t) {
        speedBoost += (targetBoost - speedBoost) * Math.min(dt * 4, 1);
        accent[0] += (targetAccent[0] - accent[0]) * Math.min(dt * 2, 1);
        accent[1] += (targetAccent[1] - accent[1]) * Math.min(dt * 2, 1);
        accent[2] += (targetAccent[2] - accent[2]) * Math.min(dt * 2, 1);

        // rotation follows reading progress (with a gentle chase)
        scrollRot += (scrollFrac * 2.5 - scrollRot) * Math.min(dt * 3, 1);

        // unwrap / rewind state
        unwrap += (unwrapTarget - unwrap) * Math.min(dt * 4, 1);
        if (unwrapTarget === 1 && unwrap > 0.985) {
            holdT += dt;
            if (holdT > 0.35) unwrapTarget = 0; // wind back into the loop
        }

        for (var i = 0; i < COUNT; i++) {
            var p = particles[i];
            p.u += 0.6 * p.sp * speedBoost * dt * (1 + unwrap * 1.5);
            if (p.u > TAU) p.u -= TAU;
        }
    }

    // --- Loop ---
    var running = false, last = 0, started = false;

    function frame(now) {
        if (!running) return;
        var t = now / 1000;
        var dt = Math.min(t - last, 0.05);
        last = t;
        step(dt, t);
        render(t);
        requestAnimationFrame(frame);
    }

    function start() {
        if (started) return;
        resize();
        if (!size) return; // hidden (mobile) — try again on resize
        started = true;

        render(0);
        mount.classList.add('bh-ready');

        if (reduceMotion) return; // static frame only

        running = true;
        last = performance.now() / 1000;
        requestAnimationFrame(frame);

        document.addEventListener('visibilitychange', function () {
            if (document.hidden) {
                running = false;
            } else if (!running) {
                running = true;
                last = performance.now() / 1000;
                requestAnimationFrame(frame);
            }
        });
    }

    window.addEventListener('resize', function () {
        if (!started) { start(); return; }
        resize();
        if (reduceMotion) render(0);
    });

    start();
})();
