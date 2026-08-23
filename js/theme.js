/*
 * Theme picker — light + three GitHub-derived dark flavors, remembered
 * across visits.
 *
 * Load this in <head> WITHOUT defer so the attribute lands before first
 * paint (no flash of the wrong theme). The toggle button + its menu inject
 * themselves once the body exists, so no page needs extra markup.
 */
(function () {
  var KEY = 'asca-theme';
  var root = document.documentElement;

  var MOON = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
  var SUN = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.2v2.2M12 19.6v2.2M4.2 12H2M22 12h-2.2M5.6 5.6 4 4M20 20l-1.6-1.6M18.4 5.6 20 4M4 20l1.6-1.6"/></svg>';

  /* mode: which structural CSS applies ([data-theme]); dot: the menu's
     swatch preview. Order here is the order the menu renders in. */
  var THEMES = [
    { key: 'light', mode: 'light', label: 'Light', dot: '#ffffff' },
    { key: 'gh-dark', mode: 'dark', label: 'GitHub Dark', dot: '#0d1117' },
    { key: 'gh-dimmed', mode: 'dark', label: 'GitHub Dark Dimmed', dot: '#22272e' },
    { key: 'gh-hc', mode: 'dark', label: 'GitHub Dark High Contrast', dot: '#0a0c10' }
  ];

  /* Dark is the house default — the OS preference does not override it;
     only an explicit choice made in the menu does. GitHub Dark Dimmed
     is the default flavor (warmer, lower-contrast than plain gh-dark). */
  var DEFAULT_KEY = 'gh-dimmed';

  function themeFor(key) {
    for (var i = 0; i < THEMES.length; i++) {
      if (THEMES[i].key === key) return THEMES[i];
    }
    return null;
  }

  function stored() {
    try {
      return localStorage.getItem(KEY);
    } catch (e) {
      return null;
    }
  }

  function save(key) {
    try {
      localStorage.setItem(KEY, key);
    } catch (e) {
      /* private mode — the choice just won't persist */
    }
  }

  function resolveInitial() {
    var v = stored();
    if (v === 'dark') return DEFAULT_KEY; /* migrate the old matte-black value */
    if (v && themeFor(v)) return v;
    return DEFAULT_KEY;
  }

  function apply(key) {
    var t = themeFor(key) || themeFor(DEFAULT_KEY);
    root.setAttribute('data-theme', t.mode);
    if (t.mode === 'dark') {
      root.setAttribute('data-flavor', t.key);
    } else {
      root.removeAttribute('data-flavor');
    }
    root.style.colorScheme = t.mode;
    return t;
  }

  var current = apply(resolveInitial());

  function mount() {
    if (document.querySelector('.theme-picker')) return;

    var wrap = document.createElement('div');
    wrap.className = 'theme-picker';

    var btn = document.createElement('button');
    btn.className = 'theme-toggle';
    btn.type = 'button';
    btn.setAttribute('aria-haspopup', 'menu');
    btn.setAttribute('aria-expanded', 'false');

    var menu = document.createElement('div');
    menu.className = 'theme-menu';
    menu.setAttribute('role', 'menu');
    menu.hidden = true;

    var items = THEMES.map(function (t) {
      var item = document.createElement('button');
      item.type = 'button';
      item.setAttribute('role', 'menuitemradio');
      item.dataset.key = t.key;

      var dot = document.createElement('span');
      dot.className = 'theme-dot';
      dot.style.background = t.dot;

      var label = document.createElement('span');
      label.className = 'theme-label';
      label.textContent = t.label;

      var check = document.createElement('span');
      check.className = 'theme-check';
      check.textContent = '✓';
      check.setAttribute('aria-hidden', 'true');

      item.appendChild(dot);
      item.appendChild(label);
      item.appendChild(check);

      item.addEventListener('click', function () {
        current = apply(t.key);
        save(t.key);
        syncButton();
        syncMenu();
        closeMenu();
        btn.focus();
      });

      menu.appendChild(item);
      return item;
    });

    function syncButton() {
      var dark = current.mode === 'dark';
      btn.innerHTML = dark ? MOON : SUN;
      var label = 'Theme: ' + current.label + '. Choose a theme';
      btn.setAttribute('aria-label', label);
      btn.title = label;
    }

    function syncMenu() {
      items.forEach(function (item) {
        item.setAttribute('aria-checked', String(item.dataset.key === current.key));
      });
    }

    function onDocClick(e) {
      if (!wrap.contains(e.target)) closeMenu();
    }

    function onKeydown(e) {
      if (e.key === 'Escape') {
        closeMenu();
        btn.focus();
      }
    }

    function openMenu() {
      menu.hidden = false;
      btn.setAttribute('aria-expanded', 'true');
      document.addEventListener('click', onDocClick, true);
      document.addEventListener('keydown', onKeydown);
    }

    function closeMenu() {
      menu.hidden = true;
      btn.setAttribute('aria-expanded', 'false');
      document.removeEventListener('click', onDocClick, true);
      document.removeEventListener('keydown', onKeydown);
    }

    btn.addEventListener('click', function () {
      if (menu.hidden) openMenu();
      else closeMenu();
    });

    syncButton();
    syncMenu();

    wrap.appendChild(btn);
    wrap.appendChild(menu);
    document.body.appendChild(wrap);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
