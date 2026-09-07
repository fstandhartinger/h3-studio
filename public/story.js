/**
 * Story mode UI.
 *
 * Kept as its own module rather than folded into app.js: it shares nothing with the
 * single-clip workspace except the session cookie, and app.js is already long enough
 * that adding a second stateful workflow to it would make both harder to follow.
 */

const $ = (sel) => document.querySelector(sel);

async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: 'same-origin', ...opts });
  const text = await res.text();
  let body = null;
  if (text) { try { body = JSON.parse(text); } catch { /* non-JSON */ } }
  if (!res.ok) throw new Error((body && (body.error || body.message)) || `HTTP ${res.status}`);
  return body;
}

const el = {
  viewClip: $('#view-clip'),
  viewStory: $('#view-story'),
  tabClip: $('#vtab-clip'),
  tabStory: $('#vtab-story'),

  premise: $('#story-premise'),
  segments: $('#story-segments'),
  seclen: $('#story-seclen'),
  style: $('#story-style'),
  language: $('#story-language'),
  width: $('#story-width'),
  height: $('#story-height'),
  steps: $('#story-steps'),
  chainOffset: $('#story-chain-offset'),

  planBtn: $('#story-plan-btn'),
  renderBtn: $('#story-render-btn'),
  cancelBtn: $('#story-cancel-btn'),
  status: $('#story-status'),

  planBox: $('#story-plan'),
  title: $('#story-title'),
  logline: $('#story-logline'),
  synopsis: $('#story-synopsis'),
  characters: $('#story-characters'),
  list: $('#story-segments-list'),
  empty: $('#story-empty'),

  finalBox: $('#story-final'),
  video: $('#story-video'),
  finalNote: $('#story-final-note'),
  download: $('#story-download'),
};

// Story mode is bolted onto a page that may not have it (older cached index.html).
if (el.viewStory && el.tabStory) init();

function init() {
  const state = { plan: null, storyId: null, es: null };

  /* ── view switching ───────────────────────────────────────────── */
  // Owns all three workspaces rather than one per module, so two modules can never both
  // think they are visible.
  const VIEWS = {
    clip:  { view: el.viewClip,  tab: el.tabClip },
    story: { view: el.viewStory, tab: el.tabStory },
    board: { view: document.querySelector('#view-board'), tab: document.querySelector('#vtab-board') },
  };

  function showView(which) {
    if (!VIEWS[which]?.view) which = 'clip';
    for (const [name, v] of Object.entries(VIEWS)) {
      if (!v.view || !v.tab) continue;
      const on = name === which;
      v.view.hidden = !on;
      v.tab.setAttribute('aria-selected', String(on));
      v.tab.tabIndex = on ? 0 : -1;
    }
    try { localStorage.setItem('h3-view', which); } catch { /* private mode */ }
  }

  for (const [name, v] of Object.entries(VIEWS)) {
    v.tab?.addEventListener('click', () => showView(name));
  }
  try {
    const saved = localStorage.getItem('h3-view');
    if (saved && saved !== 'clip') showView(saved);
  } catch { /* private mode */ }

  /* ── planning ─────────────────────────────────────────────────── */
  function setStatus(msg, kind = '') {
    el.status.textContent = msg;
    el.status.dataset.kind = kind;
  }

  el.planBtn.addEventListener('click', async () => {
    const premise = el.premise.value.trim();
    if (!premise) { setStatus('Write a premise first.', 'warn'); el.premise.focus(); return; }

    el.planBtn.disabled = true;
    el.renderBtn.disabled = true;
    setStatus('Writing the arc, the scenes and the shot prompts… this takes a minute.');
    try {
      const { plan } = await api('/api/story/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          premise,
          segments: Number(el.segments.value) || 4,
          secondsPerSegment: Number(el.seclen.value) || 5,
          style: el.style.value.trim(),
          language: el.language.value.trim(),
        }),
      });
      state.plan = plan;
      renderPlan(plan);
      setStatus(`${plan.prompts.length} shots · ${plan.totalSeconds}s total. `
              + 'Edit any prompt below, then render.', 'ok');
      el.renderBtn.disabled = false;
    } catch (e) {
      setStatus(e.message, 'error');
    } finally {
      el.planBtn.disabled = false;
    }
  });

  function renderPlan(plan) {
    el.empty.hidden = true;
    el.planBox.hidden = false;
    el.title.textContent = plan.title || 'Untitled';
    el.logline.textContent = plan.logline || '';
    el.synopsis.textContent = plan.synopsis || '';

    el.characters.replaceChildren(...(plan.characters || []).map((c) => {
      const d = document.createElement('p');
      d.className = 'note';
      const b = document.createElement('strong');
      b.textContent = `${c.name}: `;
      d.append(b, document.createTextNode(c.look || ''));
      return d;
    }));

    el.list.replaceChildren(...plan.prompts.map((p, i) => {
      const wrap = document.createElement('details');
      wrap.className = 'adv story-seg';
      wrap.id = `story-seg-${p.index}`;

      const sum = document.createElement('summary');
      sum.innerHTML = `<span class="story-seg-n">${p.index}</span> `;
      const t = document.createElement('span');
      t.textContent = p.title || `Shot ${p.index}`;
      const badge = document.createElement('span');
      badge.className = 'story-seg-state';
      badge.dataset.state = 'pending';
      badge.textContent = 'pending';
      sum.append(t, badge);

      const ta = document.createElement('textarea');
      ta.className = 'prompt';
      ta.rows = 10;
      ta.value = p.prompt;
      ta.dataset.index = String(i);

      const vid = document.createElement('video');
      vid.className = 'result-video';
      vid.controls = true;
      vid.playsInline = true;
      vid.hidden = true;

      wrap.append(sum, ta, vid);
      return wrap;
    }));
  }

  /* ── rendering ────────────────────────────────────────────────── */
  el.renderBtn.addEventListener('click', async () => {
    if (!state.plan) return;
    const prompts = [...el.list.querySelectorAll('textarea')].map((t) => t.value);

    el.renderBtn.disabled = true;
    el.planBtn.disabled = true;
    el.finalBox.hidden = true;
    setStatus('Queuing the first shot…');
    try {
      const { storyId } = await api('/api/story/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plan: state.plan,
          prompts,
          width: Number(el.width.value) || 1344,
          height: Number(el.height.value) || 768,
          steps: Number(el.steps.value) || 6,
          chainOffsetSec: Number(el.chainOffset.value),
        }),
      });
      state.storyId = storyId;
      el.cancelBtn.hidden = false;
      subscribe(storyId);
    } catch (e) {
      setStatus(e.message, 'error');
      el.renderBtn.disabled = false;
      el.planBtn.disabled = false;
    }
  });

  el.cancelBtn.addEventListener('click', async () => {
    if (!state.storyId) return;
    try { await api(`/api/story/${state.storyId}/cancel`, { method: 'POST' }); } catch { /* ignore */ }
    setStatus('Cancelling…', 'warn');
  });

  function segEls(index) {
    const wrap = document.getElementById(`story-seg-${index}`);
    return wrap
      ? { wrap, badge: wrap.querySelector('.story-seg-state'), video: wrap.querySelector('video') }
      : {};
  }

  function applySegment(s) {
    const { badge, video } = segEls(s.index);
    if (badge) {
      badge.dataset.state = s.state;
      badge.textContent = s.state;
    }
    if (video && s.videoUrl) {
      video.src = s.videoUrl;
      video.hidden = false;
    }
  }

  function subscribe(storyId) {
    state.es?.close();
    const es = new EventSource(`/api/story/${storyId}/events`);
    state.es = es;

    es.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }

      if (m.type === 'snapshot') {
        (m.story?.segments || []).forEach(applySegment);
        if (m.story?.stage) setStatus(`Rendering — ${m.story.stage}`);
        return;
      }
      if (m.type === 'segment') {
        applySegment(m);
        if (m.stage) setStatus(`Rendering — ${m.stage}`);
        return;
      }
      if (m.type === 'stage') {
        setStatus(m.stage === 'stitching' ? 'Stitching the shots together…' : m.stage);
        return;
      }
      if (m.type === 'done') {
        setStatus(`Done — ${Math.round(m.durationSec || 0)}s in ${m.elapsedSec}s (${m.method}).`, 'ok');
        el.finalBox.hidden = false;
        el.video.src = m.videoUrl;
        el.download.href = m.videoUrl;
        el.download.setAttribute('download', `${(state.plan?.title || 'story').replace(/[^\w-]+/g, '-')}.mp4`);
        el.finalNote.textContent =
          `${state.plan?.prompts?.length || 0} shots · ${Math.round(m.durationSec || 0)}s · joined by ${m.method}`;
        finish();
        return;
      }
      if (m.type === 'error') {
        setStatus(m.message, 'error');
        finish();
      }
    };
    // The server ends the stream on a terminal event; that surfaces here as onerror.
    es.onerror = () => { es.close(); };
  }

  function finish() {
    state.es?.close();
    state.es = null;
    el.cancelBtn.hidden = true;
    el.renderBtn.disabled = false;
    el.planBtn.disabled = false;
  }
}
