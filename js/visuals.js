
// Visuals.js - 3D Sci-Fi Element
// Inspired by: Black Holes, Wormholes, and Time

document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('visual-canvas');
    if (!container) return;

    // --- Scene Setup ---
    const scene = new THREE.Scene();
    // Fog for depth fading - matching the page background roughly or dark
    scene.fog = new THREE.FogExp2(0x000000, 0.002);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    // Position camera to look at the object
    camera.position.z = 20;
    camera.position.y = 5;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    // Use container dimensions, not window
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // --- Interaction: Click to Scroll Top ---
    container.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // --- Object Creation: "The Singularity" (Möbius-like Particle System) ---
    // We'll create a twisted torus knot represented by particles

    const particlesGeometry = new THREE.BufferGeometry();
    const particleCount = 2500; // Reduced count for smaller view

    const positions = [];
    const colors = [];
    const sizes = [];

    const color1 = new THREE.Color(0x6e46be); // User's purple
    const color2 = new THREE.Color(0x000000); // Black
    const color3 = new THREE.Color(0x4444ff); // Sci-fi Blue

    for (let i = 0; i < particleCount; i++) {
        // Parametric equation for a torus knot/möbius strip feel
        // p and q integers determine the knot shape (e.g., 2,3 is a trefoil)
        // Let's try something flowing.
        const t = (i / particleCount) * Math.PI * 20;

        // Torus Knot logic
        // x = (R + r * cos(q*t)) * cos(p*t)
        // y = (R + r * cos(q*t)) * sin(p*t)
        // z = r * sin(q*t)

        const p = 2; // loops around axis
        const q = 3; // loops through hole
        const radius = 5; // Slightly compacted
        const tube = 2.0;

        // Introduce some randomness/noise to make it look like a cloud/energy
        const spread = (Math.random() - 0.5) * 1.5;

        const x = (radius + (tube + spread) * Math.cos(q * t)) * Math.cos(p * t);
        const y = (radius + (tube + spread) * Math.cos(q * t)) * Math.sin(p * t);
        const z = (tube + spread) * Math.sin(q * t);

        positions.push(x, y, z);

        // Color gradient based on position
        const mixedColor = color1.clone().lerp(color3, Math.random());
        colors.push(mixedColor.r, mixedColor.g, mixedColor.b);

        sizes.push(Math.random() * 0.25); // Slightly larger relative to small canvas
    }

    particlesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    particlesGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    particlesGeometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1)); // We'll need a shader for size or just use PointsMaterial


    // Fun helper to create a soft glowing circle texture
    function getTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        const context = canvas.getContext('2d');

        // Radial gradient for soft sphere look
        const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
        gradient.addColorStop(0, 'rgba(255,255,255,1)');
        gradient.addColorStop(0.4, 'rgba(255,255,255,0.8)');
        gradient.addColorStop(1, 'rgba(0,0,0,0)');

        context.fillStyle = gradient;
        context.fillRect(0, 0, 32, 32);

        const texture = new THREE.Texture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    const particlesMaterial = new THREE.PointsMaterial({
        size: 0.4, // Visual size bump for small screen
        map: getTexture(),
        vertexColors: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        opacity: 0.8 // Brighter for small icon
    });

    const particleSystem = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particleSystem);

    // --- Interaction vars ---
    let mouseX = 0;
    let mouseY = 0;

    // --- Mouse Interaction ---
    // Track mouse position relative to window center
    window.addEventListener('mousemove', (event) => {
        mouseX = (event.clientX / window.innerWidth) * 2 - 1;
        mouseY = -(event.clientY / window.innerHeight) * 2 + 1;
    });

    // --- Animation Vars ---
    let scrollY = 0;
    let currentSection = 0;

    // Updated Brand Colors from CSS
    // Intro: #6e46be (Purple - Links/Brand)
    // Experience: #1772d0 (Blue - Links/Professional)
    // Research: #f09228 (Orange - Hover/Highlight)
    // Reflections: #222222 (Dark - Text/Core) or maybe a lighter variant for visibility
    const sectionColors = [
        new THREE.Color(0x6e46be), // Intro
        new THREE.Color(0x1772d0), // Experience
        new THREE.Color(0xf09228), // Research
        new THREE.Color(0x444444)  // Reflections (Dark Grey)
    ];

    let targetColor = sectionColors[0];
    let currentColorVal = sectionColors[0].clone();

    // Listen to scroll
    window.addEventListener('scroll', () => {
        scrollY = window.scrollY;

        // Simple section detection
        const docHeight = document.body.scrollHeight - window.innerHeight;
        const scrollPerc = scrollY / Math.max(docHeight, 1);

        // Map scroll percentage to roughly 4 sections
        let index = Math.floor(scrollPerc * 4);
        if (index > 3) index = 3;
        if (index < 0) index = 0;

        if (index !== currentSection) {
            currentSection = index;
            targetColor = sectionColors[index];
        }
    });

    // --- Animation Loop ---
    const clock = new THREE.Clock();
    let previousTime = 0;

    function animate() {
        requestAnimationFrame(animate);

        const elapsedTime = clock.getElapsedTime();
        const deltaTime = elapsedTime - previousTime;
        previousTime = elapsedTime;

        // Color Update
        currentColorVal.lerp(targetColor, deltaTime * 2.0); // Speed of color change
        particlesMaterial.color.set(currentColorVal);


        // Rotation - Indication of "Life"
        // Base rotation
        particleSystem.rotation.x = elapsedTime * 0.2;
        particleSystem.rotation.y = elapsedTime * 0.3;

        // Mouse Interaction (Parallax / Tilt)
        // Add mouse influence to rotation
        particleSystem.rotation.x += mouseY * 0.5;
        particleSystem.rotation.y += mouseX * 0.5;

        // Scroll Influence - "Progress Meter" feel
        // As you scroll, the Z rotation aligns
        const scrollFactor = scrollY * 0.002;
        particleSystem.rotation.z = scrollFactor;

        // Compact Scale
        // Pulse
        const scale = 0.8 + Math.sin(elapsedTime * 2.0) * 0.05;
        particleSystem.scale.set(scale, scale, scale);

        renderer.render(scene, camera);
    }

    animate();

    // --- Resize Handler ---
    window.addEventListener('resize', () => {
        // Update to container size, not window
        const width = container.clientWidth;
        const height = container.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    });
});
