/* =====================================================================
   TUD-Station Showcase — Navigation & Interaction Script
   ===================================================================== */

(function () {
  'use strict';

  /* ── SIDEBAR TOGGLE (mobile) ──────────────────────────────────────── */
  const sidebar   = document.getElementById('sidebar');
  const toggle    = document.getElementById('nav-toggle');
  const overlay   = document.createElement('div');

  overlay.id = 'nav-overlay';
  Object.assign(overlay.style, {
    display: 'none', position: 'fixed', inset: '0',
    background: 'rgba(0,0,0,.4)', zIndex: '150'
  });
  document.body.appendChild(overlay);

  function openSidebar() {
    sidebar.classList.add('open');
    overlay.style.display = 'block';
    toggle.textContent = '✕';
  }

  function closeSidebar() {
    sidebar.classList.remove('open');
    overlay.style.display = 'none';
    toggle.textContent = '☰';
  }

  if (toggle) {
    toggle.textContent = '☰';
    toggle.addEventListener('click', () => {
      sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
    });
    overlay.addEventListener('click', closeSidebar);
  }

  /* ── ACTIVE NAV LINK ──────────────────────────────────────────────── */
  const currentPage = location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('#sidebar nav a').forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPage || (currentPage === '' && href === 'index.html')) {
      link.classList.add('active');
    }
  });

  /* ── SCROLL SPY (in-page sections) ───────────────────────────────── */
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('#sidebar nav a[href^="#"]');

  if (sections.length && navLinks.length) {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          navLinks.forEach(l => l.classList.remove('active'));
          const active = document.querySelector(`#sidebar nav a[href="#${e.target.id}"]`);
          if (active) active.classList.add('active');
        }
      });
    }, { rootMargin: '-30% 0px -60% 0px', threshold: 0 });

    sections.forEach(s => obs.observe(s));
  }

  /* ── SCROLL REVEAL ────────────────────────────────────────────────── */
  const revObs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        revObs.unobserve(e.target);
      }
    });
  }, { threshold: 0.06 });

  document.querySelectorAll('.reveal').forEach(el => revObs.observe(el));

  /* ── COUNTER ANIMATION ────────────────────────────────────────────── */
  function animateCount(el) {
    const target = parseFloat(el.dataset.target);
    const suffix = el.dataset.suffix || '';
    const duration = 1600;
    const isFloat = String(target).includes('.');
    const decimals = isFloat ? (String(target).split('.')[1] || '').length : 0;
    const start = performance.now();

    function step(now) {
      const t = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      const val = target * ease;
      el.textContent = (isFloat ? val.toFixed(decimals) : Math.round(val).toLocaleString()) + suffix;
      if (t < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  }

  const countObs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        animateCount(e.target);
        countObs.unobserve(e.target);
      }
    });
  }, { threshold: 0.3 });

  document.querySelectorAll('.count-up[data-target]').forEach(el => countObs.observe(el));

  /* ── TIMELINE ACCORDION ───────────────────────────────────────────── */
  document.querySelectorAll('.tl-item .tl-header').forEach(header => {
    const body = header.nextElementSibling;
    if (!body || !body.classList.contains('tl-body')) return;

    // Start collapsed
    body.style.display = 'none';
    header.style.cursor = 'pointer';

    const indicator = document.createElement('span');
    indicator.textContent = '▸';
    indicator.style.cssText = 'font-size:12px; color:var(--accent); transition:transform .2s; display:inline-block; margin-left:auto;';
    header.appendChild(indicator);

    header.addEventListener('click', () => {
      const open = body.style.display !== 'none';
      body.style.display = open ? 'none' : 'block';
      indicator.style.transform = open ? '' : 'rotate(90deg)';
    });
  });

  /* ── LIGHTBOX ─────────────────────────────────────────────────────── */
  let lb = document.getElementById('lightbox');

  if (!lb) {
    lb = document.createElement('div');
    lb.id = 'lightbox';
    lb.innerHTML = '<span class="lb-close">✕</span><img src="" alt="Preview">';
    document.body.appendChild(lb);
  }

  const lbImg   = lb.querySelector('img');
  const lbClose = lb.querySelector('.lb-close');

  function openLightbox(src, alt) {
    lbImg.src = src;
    lbImg.alt = alt || '';
    lb.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeLightbox() {
    lb.classList.remove('open');
    document.body.style.overflow = '';
    lbImg.src = '';
  }

  lbClose.addEventListener('click', closeLightbox);
  lb.addEventListener('click', e => { if (e.target === lb) closeLightbox(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

  document.querySelectorAll('.img-card').forEach(card => {
    card.addEventListener('click', () => {
      const img     = card.querySelector('img');
      const caption = card.querySelector('.img-caption');
      if (img) openLightbox(img.src, caption ? caption.textContent : '');
    });
  });

  /* ── IFRAME HEIGHT GUARD ──────────────────────────────────────────── */
  document.querySelectorAll('.iframe-wrap iframe').forEach(frame => {
    frame.addEventListener('load', () => {
      // Ensure iframe is visible after load
      frame.style.opacity = '1';
    });
    frame.style.opacity = '0';
    frame.style.transition = 'opacity .3s';
  });

})();
