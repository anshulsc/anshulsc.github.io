/*
 * Theme switch — light / dark, remembered across visits.
 *
 * Load this in <head> WITHOUT defer so the attribute lands before first paint
 * (no white flash on a dark-mode visit). The toggle button injects itself once
 * the body exists, so no page needs extra markup.
 */
(function () {
  var KEY = 'asca-theme';
  var root = document.documentElement;

  var MOON = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
  var SUN = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.2v2.2M12 19.6v2.2M4.2 12H2M22 12h-2.2M5.6 5.6 4 4M20 20l-1.6-1.6M18.4 5.6 20 4M4 20l1.6-1.6"/></svg>';

  function stored() {
    try {
      return localStorage.getItem(KEY);
    } catch (e) {
      return null;
    }
  }

  function apply(theme) {
    root.setAttribute('data-theme', theme);
    root.style.colorScheme = theme;
  }

  /* Dark is the house default — the OS preference does not override it;
     only an explicit choice made with the toggle does. */
  apply(stored() || 'dark');

  function mount() {
    if (document.querySelector('.theme-toggle')) return;

    var btn = document.createElement('button');
    btn.className = 'theme-toggle';
    btn.type = 'button';

    var sync = function () {
      var dark = root.getAttribute('data-theme') === 'dark';
      var label = dark ? 'Switch to light mode' : 'Switch to dark mode';
      btn.innerHTML = dark ? SUN : MOON;
      btn.setAttribute('aria-label', label);
      btn.setAttribute('title', label);
    };

    btn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      try {
        localStorage.setItem(KEY, next);
      } catch (e) {
        /* private mode — the choice just won't persist */
      }
      apply(next);
      sync();
    });

    sync();
    document.body.appendChild(btn);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
