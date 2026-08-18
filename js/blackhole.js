/**
 * The Singularity — a small black-hole ornament.
 * Inspired by: Black Holes, Wormholes, and Time.
 *
 * Pure canvas 2D, no dependencies. A tilted accretion disk of particles
 * on Keplerian orbits around a shadow core with a photon ring and faint
 * lensed arcs. The accent color follows the section being read
 * (intro → experience → research → reflections). Click = back to top.
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

    window.addEventListener('scroll', function () {
        var doc = document.documentElement;
        var max = Math.max(doc.scrollHeight - window.innerHeight, 1);
        var idx = Math.min(3, Math.max(0, Math.floor((window.scrollY / max) * 4)));
        targetAccent = SECTION_COLORS[idx];
    }, { passive: true });

    // --- Disk geometry (normalized to R) ---
    var SHADOW_R = 0.22;          // event-horizon shadow
    var PHOTON_R = SHADOW_R * 1.22;
    var DISK_IN = 0.34;
    var DISK_OUT = 0.92;
    var FLAT = 0.30;              // disk tilt (1 = face-on)
    var ROLL = -0.38;             // slight roll for character

    // --- Particles ---
    var COUNT = 240;
    var particles = [];
    for (var i = 0; i < COUNT; i++) {
        particles.push({
            r: DISK_IN + Math.pow(Math.random(), 0.7) * (DISK_OUT - DISK_IN),
            th: Math.random() * Math.PI * 2,
            sz: 0.55 + Math.random() * 1.1,
            tw: Math.random() * Math.PI * 2
        });
    }

    // --- Interaction ---
    var speedBoost = 1, targetBoost = 1;
    mount.addEventListener('mouseenter', function () { targetBoost = 1.8; });
    mount.addEventListener('mouseleave', function () { targetBoost = 1; });

    function backToTop() { window.scrollTo({ top: 0, behavior: 'smooth' }); }
    mount.addEventListener('click', backToTop);
    mount.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); backToTop(); }
    });

    // --- Drawing ---
    function drawParticle(p, t) {
        var x = Math.cos(p.th) * p.r * R;
        var y = Math.sin(p.th) * p.r * R * FLAT;

        // Doppler-ish beaming: approaching side glows stronger
        var beam = 0.45 + 0.55 * (0.5 + 0.5 * Math.cos(p.th));
        var twinkle = 0.85 + 0.15 * Math.sin(t * 3 + p.tw);
        var heat = 1 - (p.r - DISK_IN) / (DISK_OUT - DISK_IN); // inner = hotter
        var a = (0.28 + 0.62 * beam * heat) * twinkle;

        // on a light page, "hotter" reads as darker & denser, not whiter
        var w = heat * heat * 0.75;
        var cr = accent[0] * (1 - w * 0.8);
        var cg = accent[1] * (1 - w * 0.8);
        var cb = accent[2] * (1 - w * 0.8);

        ctx.fillStyle = 'rgba(' + (cr | 0) + ',' + (cg | 0) + ',' + (cb | 0) + ',' + a.toFixed(3) + ')';
        ctx.beginPath();
        ctx.arc(x, y, p.sz * dpr * (0.7 + 0.5 * beam), 0, Math.PI * 2);
        ctx.fill();
    }

    function render(t) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.translate(R, R);
        ctx.rotate(ROLL);

        // soft ambient glow
        var glow = ctx.createRadialGradient(0, 0, R * SHADOW_R, 0, 0, R);
        glow.addColorStop(0, 'rgba(' + accent[0] + ',' + accent[1] + ',' + accent[2] + ',0.10)');
        glow.addColorStop(1, 'rgba(' + accent[0] + ',' + accent[1] + ',' + accent[2] + ',0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(0, 0, R, 0, Math.PI * 2);
        ctx.fill();

        // lensed arcs — the far side of the disk bent over and under the shadow
        var dark = [(accent[0] * 0.6) | 0, (accent[1] * 0.6) | 0, (accent[2] * 0.6) | 0];
        ctx.lineWidth = 1.6 * dpr;
        ctx.strokeStyle = 'rgba(' + dark[0] + ',' + dark[1] + ',' + dark[2] + ',0.50)';
        ctx.beginPath();
        ctx.arc(0, 0, R * PHOTON_R * 1.35, Math.PI * 1.05, Math.PI * 1.95);
        ctx.stroke();
        ctx.strokeStyle = 'rgba(' + dark[0] + ',' + dark[1] + ',' + dark[2] + ',0.24)';
        ctx.beginPath();
        ctx.arc(0, 0, R * PHOTON_R * 1.35, Math.PI * 0.08, Math.PI * 0.92);
        ctx.stroke();

        // far half of the disk (passes behind the core)
        var i, p;
        for (i = 0; i < COUNT; i++) {
            p = particles[i];
            if (Math.sin(p.th) < 0) drawParticle(p, t);
        }

        // shadow core
        ctx.fillStyle = '#0b0b0b';
        ctx.beginPath();
        ctx.arc(0, 0, R * SHADOW_R, 0, Math.PI * 2);
        ctx.fill();

        // photon ring — thin, saturated, hugging the shadow
        ctx.lineWidth = 1.5 * dpr;
        ctx.strokeStyle = 'rgba(' + dark[0] + ',' + dark[1] + ',' + dark[2] + ',0.95)';
        ctx.beginPath();
        ctx.arc(0, 0, R * PHOTON_R, 0, Math.PI * 2);
        ctx.stroke();

        // near half of the disk (passes in front of the core)
        for (i = 0; i < COUNT; i++) {
            p = particles[i];
            if (Math.sin(p.th) >= 0) drawParticle(p, t);
        }

        ctx.restore();
    }

    // --- Simulation ---
    function step(dt, t) {
        speedBoost += (targetBoost - speedBoost) * Math.min(dt * 4, 1);
        accent[0] += (targetAccent[0] - accent[0]) * Math.min(dt * 2, 1);
        accent[1] += (targetAccent[1] - accent[1]) * Math.min(dt * 2, 1);
        accent[2] += (targetAccent[2] - accent[2]) * Math.min(dt * 2, 1);

        for (var i = 0; i < COUNT; i++) {
            var p = particles[i];
            // Keplerian: inner orbits are faster (ω ∝ r^-3/2)
            p.th += (0.55 / Math.pow(p.r, 1.5)) * dt * speedBoost;
            // slow inspiral, respawn at the outer edge
            p.r -= 0.006 * (DISK_IN / p.r) * dt;
            if (p.r < DISK_IN) {
                p.r = DISK_OUT;
                p.th = Math.random() * Math.PI * 2;
            }
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

        // settle particle phases so the first frame already looks alive
        for (var k = 0; k < 90; k++) step(1 / 60, 0);
        render(0);
        mount.classList.add('bh-ready');

        if (reduceMotion) return; // static frame only

        running = true;
        last = performance.now() / 1000;
        requestAnimationFrame(frame);

        document.addEventListener('visibilitychange', function () {
            if (document.hidden) {
                running = false;
            } else if (!running && !reduceMotion) {
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
