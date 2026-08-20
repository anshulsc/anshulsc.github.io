/*
 * Passphrase gate for the locked notes series.
 *
 * This is a courtesy lock, not security: the page content still ships to the
 * browser and is visible in view-source / devtools. It keeps casual visitors
 * out of the drafts, nothing more. Do not put anything genuinely private here.
 *
 * Unlocking is remembered in localStorage and covers every gated page.
 */
(function () {
  var KEY = 'asca-notes-unlocked';
  var HASH = 'e0f895872d65b2528feec97350a3a212b3d4ab88748e25d022a34641d338216b';

  if (typeof localStorage !== 'undefined' && localStorage.getItem(KEY) === HASH) return;

  var STYLE = [
    'body.gate-lock > *:not(.gate){visibility:hidden !important}',
    'body.gate-lock{overflow:hidden !important}',
    '.gate{position:fixed;inset:0;z-index:9999;display:flex;align-items:center;',
    'justify-content:center;padding:24px;background:#0a0a0a;color:#e6e6e6;',
    'font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}',
    '.gate-box{width:100%;max-width:380px;text-align:center}',
    '.gate-mark{width:64px;height:64px;border-radius:50%;margin:0 auto 22px;display:block;background:#1c172b;border:1px solid rgba(176,136,249,.3)}',
    '.gate h1{font-size:1.15rem;font-weight:600;margin:0 0 8px;color:#f5f5f5}',
    '.gate p{font-size:.85rem;line-height:1.6;color:#a0a0a0;margin:0 0 22px}',
    '.gate form{display:flex;gap:8px}',
    '.gate input{flex:1;padding:10px 14px;border-radius:8px;border:1px solid #272727;',
    'background:#131313;color:#e6e6e6;font-size:.9rem;font-family:inherit}',
    '.gate input:focus{outline:none;border-color:#b088f9}',
    '.gate button{padding:10px 18px;border-radius:8px;border:1px solid #b088f9;',
    'background:transparent;color:#b088f9;font-size:.85rem;font-weight:600;cursor:pointer;font-family:inherit}',
    '.gate button:hover{background:#b088f9;color:#0a0a0a}',
    '.gate .gate-msg{min-height:20px;margin:14px 0 0;font-size:.8rem;color:#f2777a}',
    '.gate .gate-back{display:inline-block;margin-top:18px;font-size:.8rem;color:#6f6f6f;text-decoration:none}',
    '.gate .gate-back:hover{color:#b088f9}'
  ].join('');

  function sha256(text) {
    if (window.crypto && window.crypto.subtle && window.TextEncoder) {
      return window.crypto.subtle
        .digest('SHA-256', new TextEncoder().encode(text))
        .then(function (buf) {
          return Array.prototype.map
            .call(new Uint8Array(buf), function (b) {
              return ('0' + b.toString(16)).slice(-2);
            })
            .join('');
        });
    }
    return Promise.resolve(null);
  }

  function depth() {
    /* how far this page sits below the site root, for the "back" link */
    var parts = location.pathname.split('/').filter(Boolean);
    return new Array(Math.max(parts.length - 1, 0) + 1).join('../') || './';
  }

  function mount() {
    var style = document.createElement('style');
    style.textContent = STYLE;
    document.head.appendChild(style);

    var gate = document.createElement('div');
    gate.className = 'gate';
    gate.innerHTML =
      '<div class="gate-box">' +
      '<img class="gate-mark" src="' + depth() + 'images/avatar.png" alt="">' +
      '<h1>These notes are locked</h1>' +
      '<p>Course notes in progress. Drop in the passphrase to read on.</p>' +
      '<form><input type="password" autocomplete="current-password" ' +
      'placeholder="passphrase" aria-label="Passphrase" autofocus>' +
      '<button type="submit">Unlock</button></form>' +
      '<p class="gate-msg" role="status"></p>' +
      '<a class="gate-back" href="' + depth() + 'Blogs/blog-index.html">← back to writing</a>' +
      '</div>';

    document.body.classList.add('gate-lock');
    document.body.appendChild(gate);

    var form = gate.querySelector('form');
    var input = gate.querySelector('input');
    var msg = gate.querySelector('.gate-msg');
    input.focus();

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      sha256(input.value).then(function (hex) {
        var ok = hex ? hex === HASH : input.value === 'knowledge';
        if (!ok) {
          msg.textContent = 'Not quite. Try again.';
          input.select();
          return;
        }
        try {
          localStorage.setItem(KEY, HASH);
        } catch (err) {
          /* private mode: unlocked for this page view only */
        }
        document.body.classList.remove('gate-lock');
        gate.remove();
      });
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
