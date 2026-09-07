/**
 * Storyboard UI.
 *
 * The workflow this supports, in order: write the board, draw the keyframes, fix the ones
 * that came out wrong, then render video. The point of the middle step is that keyframes
 * are cheap (~20 s each) and video is not (~2 min a shot), so every disagreement with the
 * model should be settled while it is still an image.
 *
 * Three ways to fix a frame, cheapest first: re-roll on a new seed, edit the prompt by
 * hand, or ask the LLM to revise the prompt from a plain-language note. Plus upload, for
 * when none of that gets there.
 */
const $ = (s) => document.querySelector(s);

async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: 'same-origin', ...opts });
  const text = await res.text();
  let body = null;
  if (text) { try { body = JSON.parse(text); } catch { /* non-JSON */ } }
  if (!res.ok) throw new Error((body && (body.error || body.message)) || `HTTP ${res.status}`);
  return body;
}

const el = {
  view: $('#view-board'),
  tab: $('#vtab-board'),
  premise: $('#board-premise'),
  keyframes: $('#board-keyframes'),
  seclen: $('#board-seclen'),
  maths: $('#board-maths'),
  style: $('#board-style'),
  language: $('#board-language'),
  width: $('#board-width'),
  height: $('#board-height'),
  imgSteps: $('#board-img-steps'),
  cfg: $('#board-cfg'),
  steps: $('#board-steps'),
  planBtn: $('#board-plan-btn'),
  framesBtn: $('#board-frames-btn'),
  renderBtn: $('#board-render-btn'),
  status: $('#board-status'),
  noChroma: $('#board-nochroma'),
  planBox: $('#board-plan'),
  title: $('#board-title'),
  logline: $('#board-logline'),
  synopsis: $('#board-synopsis'),
  characters: $('#board-characters'),
  grid: $('#board-kf-grid'),
  shots: $('#board-shot-list'),
  empty: $('#board-empty'),
  finalBox: $('#board-final'),
  video: $('#board-video'),
  finalNote: $('#board-final-note'),
  download: $('#board-download'),
};

if (el.view && el.tab) init();

function init() {
  // keyframe index (1-based) -> { imageId, url, comfyName, busy }
  const drawn = new Map();
  let board = null;
  let es = null;

  const setStatus = (m, kind = '') => {
    el.status.textContent = m;
    el.status.dataset.kind = kind;
  };

  function recalcMaths() {
    const k = Math.max(2, Number(el.keyframes.value) || 2);
    const secs = Number(el.seclen.value) || 5;
    el.maths.textContent = `${k} keyframes → ${k - 1} shots → ${Math.round((k - 1) * secs)} s`;
  }
  el.keyframes.addEventListener('input', recalcMaths);
  el.seclen.addEventListener('input', recalcMaths);
  recalcMaths();

  // Chroma lives on the pod, so its availability changes when a pod comes and goes.
  async function checkChroma() {
    try {
      const st = await api('/api/status');
      const ok = !!st?.image?.available;
      el.noChroma.hidden = ok;
      return ok;
    } catch { return false; }
  }
  checkChroma();
  setInterval(checkChroma, 30000);

  /* ── 1. plan ──────────────────────────────────────────────────── */
  el.planBtn.addEventListener('click', async () => {
    const premise = el.premise.value.trim();
    if (!premise) { setStatus('Write a premise first.', 'warn'); el.premise.focus(); return; }
    el.planBtn.disabled = true;
    el.framesBtn.disabled = true;
    el.renderBtn.disabled = true;
    setStatus('Writing the arc, the keyframes and the shot prompts…');
    try {
      const { board: b } = await api('/api/storyboard/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          premise,
          keyframes: Number(el.keyframes.value) || 5,
          secondsPerSegment: Number(el.seclen.value) || 5,
          style: el.style.value.trim(),
          language: el.language.value.trim(),
        }),
      });
      board = b;
      drawn.clear();
      paintBoard(b);
      setStatus(`${b.keyframes.length} keyframes → ${b.shots.length} shots → ${b.totalSeconds}s. `
              + 'Draw the keyframes next.', 'ok');
      el.framesBtn.disabled = false;
    } catch (e) {
      setStatus(e.message, 'error');
    } finally {
      el.planBtn.disabled = false;
    }
  });

  function paintBoard(b) {
    el.empty.hidden = true;
    el.planBox.hidden = false;
    el.finalBox.hidden = true;
    el.title.textContent = b.title || 'Untitled';
    el.logline.textContent = b.logline || '';
    el.synopsis.textContent = b.synopsis || '';

    el.characters.replaceChildren(...(b.characters || []).map((c) => {
      const p = document.createElement('p');
      p.className = 'note';
      const s = document.createElement('strong');
      s.textContent = `${c.name}: `;
      p.append(s, document.createTextNode(c.look || ''));
      return p;
    }));

    el.grid.replaceChildren(...b.keyframes.map((k) => keyframeCard(k)));
    el.shots.replaceChildren(...b.shots.map((sh) => {
      const d = document.createElement('details');
      d.className = 'adv story-seg';
      d.id = `board-shot-${sh.index}`;
      const sum = document.createElement('summary');
      const n = document.createElement('span');
      n.className = 'story-seg-n';
      n.textContent = sh.index;
      const t = document.createElement('span');
      t.textContent = sh.title || `Shot ${sh.index}`;
      const badge = document.createElement('span');
      badge.className = 'story-seg-state';
      badge.dataset.state = 'pending';
      badge.textContent = 'pending';
      sum.append(n, t, badge);
      const ta = document.createElement('textarea');
      ta.className = 'prompt';
      ta.rows = 9;
      ta.value = sh.prompt;
      const v = document.createElement('video');
      v.className = 'result-video';
      v.controls = true;
      v.playsInline = true;
      v.hidden = true;
      d.append(sum, ta, v);
      return d;
    }));
  }

  /* ── 2. keyframes ─────────────────────────────────────────────── */
  function keyframeCard(k) {
    const card = document.createElement('div');
    card.className = 'kf-card';
    card.id = `kf-${k.index}`;

    const head = document.createElement('div');
    head.className = 'kf-head';
    const n = document.createElement('span');
    n.className = 'story-seg-n';
    n.textContent = k.index;
    const t = document.createElement('span');
    t.className = 'kf-title';
    t.textContent = k.title || `Keyframe ${k.index}`;
    const state = document.createElement('span');
    state.className = 'story-seg-state';
    state.dataset.state = 'pending';
    state.textContent = 'not drawn';
    head.append(n, t, state);

    const img = document.createElement('img');
    img.className = 'kf-img';
    img.alt = `Keyframe ${k.index}: ${k.moment || ''}`;
    img.loading = 'lazy';
    img.hidden = true;

    const ph = document.createElement('div');
    ph.className = 'kf-placeholder';
    ph.textContent = k.moment || '';

    const promptTa = document.createElement('textarea');
    promptTa.className = 'prompt kf-prompt';
    promptTa.rows = 5;
    promptTa.value = k.imagePrompt;

    const negTa = document.createElement('input');
    negTa.type = 'text';
    negTa.className = 'kf-neg';
    negTa.value = k.negative || '';
    negTa.setAttribute('aria-label', `Negative prompt for keyframe ${k.index}`);

    const noteIn = document.createElement('input');
    noteIn.type = 'text';
    noteIn.className = 'kf-note';
    noteIn.placeholder = 'change it in words, e.g. "wider shot, angrier"';
    noteIn.setAttribute('aria-label', `Revision note for keyframe ${k.index}`);

    const row = document.createElement('div');
    row.className = 'kf-actions';
    const mk = (label, title, fn) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'btn btn-ghost btn-sm';
      b.textContent = label;
      b.title = title;
      b.addEventListener('click', fn);
      return b;
    };
    const upload = document.createElement('input');
    upload.type = 'file';
    upload.accept = 'image/*';
    upload.hidden = true;

    const draw = (seed) => drawKeyframe(k.index, { seed });
    row.append(
      mk('Draw', 'Generate this keyframe', () => draw(null)),
      mk('Re-roll', 'Same prompt, a new seed', () => draw(null)),
      mk('Revise', 'Rewrite the prompt from the note, then redraw', async () => {
        const note = noteIn.value.trim();
        if (!note) { setStatus('Write what to change first.', 'warn'); noteIn.focus(); return; }
        setStatus(`Revising keyframe ${k.index}…`);
        try {
          const bible = (board.characters || []).map((c) => `- ${c.name}: ${c.look}`).join('\n');
          const { prompt } = await api('/api/storyboard/revise-prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ imagePrompt: promptTa.value, note, style: board.style, bible }),
          });
          promptTa.value = prompt;
          noteIn.value = '';
          setStatus(`Keyframe ${k.index} prompt revised. Drawing…`);
          await draw(null);
        } catch (e) { setStatus(e.message, 'error'); }
      }),
      mk('Replace…', 'Upload your own image instead', () => upload.click()),
    );

    upload.addEventListener('change', async () => {
      const f = upload.files?.[0];
      if (!f) return;
      setStatus(`Uploading a replacement for keyframe ${k.index}…`);
      try {
        const fd = new FormData();
        fd.append('file', f);
        const r = await api('/api/upload', { method: 'POST', body: fd });
        drawn.set(k.index, { comfyName: r.name, url: r.previewUrl, uploaded: true });
        img.src = r.previewUrl;
        img.hidden = false;
        ph.hidden = true;
        state.dataset.state = 'done';
        state.textContent = 'replaced';
        setStatus(`Keyframe ${k.index} replaced.`, 'ok');
        refreshRenderButton();
      } catch (e) { setStatus(e.message, 'error'); }
      upload.value = '';
    });

    card.append(head, ph, img, promptTa, negTa, noteIn, row, upload);
    card._parts = { img, ph, state, promptTa, negTa };
    return card;
  }

  async function drawKeyframe(index, { seed = null } = {}) {
    const card = document.getElementById(`kf-${index}`);
    if (!card) return;
    const { img, ph, state, promptTa, negTa } = card._parts;
    state.dataset.state = 'running';
    state.textContent = 'drawing';
    try {
      const r = await api('/api/image/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: promptTa.value,
          negative: negTa.value,
          width: Number(el.width.value) || 1344,
          height: Number(el.height.value) || 768,
          steps: Number(el.imgSteps.value) || 26,
          cfg: Number(el.cfg.value) || 4,
          seed,
        }),
      });
      // The video graph loads keyframes by filename from ComfyUI's input/ directory, so
      // the generated PNG has to be pushed back there before it can be used as one.
      const put = await api(`/api/image/${r.imageId}/as-input`, { method: 'POST' });
      drawn.set(index, { imageId: r.imageId, url: r.url, comfyName: put.name });
      img.src = `${r.url}?v=${Date.now()}`;
      img.hidden = false;
      ph.hidden = true;
      state.dataset.state = 'done';
      state.textContent = `seed ${r.seed}`;
      refreshRenderButton();
    } catch (e) {
      state.dataset.state = 'error';
      state.textContent = 'failed';
      setStatus(`Keyframe ${index}: ${e.message}`, 'error');
      throw e;
    }
  }

  function refreshRenderButton() {
    const ready = board && board.keyframes.every((k) => drawn.get(k.index)?.comfyName);
    el.renderBtn.disabled = !ready;
    if (ready) setStatus('All keyframes drawn. Render the shots when you are happy with them.', 'ok');
  }

  el.framesBtn.addEventListener('click', async () => {
    if (!board) return;
    el.framesBtn.disabled = true;
    el.planBtn.disabled = true;
    try {
      // Sequential, not parallel: one GPU, and a queue of eight images just makes the
      // first one land later without making the last one land sooner.
      for (const k of board.keyframes) {
        if (drawn.get(k.index)?.comfyName) continue;
        setStatus(`Drawing keyframe ${k.index} of ${board.keyframes.length}…`);
        await drawKeyframe(k.index, {});
      }
      setStatus('All keyframes drawn. Fix any you dislike, then render.', 'ok');
    } catch { /* per-frame error already reported */ } finally {
      el.framesBtn.disabled = false;
      el.planBtn.disabled = false;
      refreshRenderButton();
    }
  });

  /* ── 3. render ────────────────────────────────────────────────── */
  el.renderBtn.addEventListener('click', async () => {
    if (!board) return;
    const names = board.keyframes.map((k) => drawn.get(k.index)?.comfyName);
    if (names.some((n) => !n)) { setStatus('A keyframe is still missing.', 'error'); return; }
    const prompts = [...el.shots.querySelectorAll('textarea')].map((t) => t.value);

    el.renderBtn.disabled = true;
    el.planBtn.disabled = true;
    el.framesBtn.disabled = true;
    setStatus('Queuing the first shot…');
    try {
      const { storyId } = await api('/api/storyboard/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          board, keyframeNames: names, prompts,
          width: Number(el.width.value) || 1344,
          height: Number(el.height.value) || 768,
          steps: Number(el.steps.value) || 6,
        }),
      });
      follow(storyId);
    } catch (e) {
      setStatus(e.message, 'error');
      el.renderBtn.disabled = false;
      el.planBtn.disabled = false;
      el.framesBtn.disabled = false;
    }
  });

  function shotEls(i) {
    const w = document.getElementById(`board-shot-${i}`);
    return w ? { badge: w.querySelector('.story-seg-state'), video: w.querySelector('video') } : {};
  }

  function applyShot(s) {
    const { badge, video } = shotEls(s.index);
    if (badge) { badge.dataset.state = s.state; badge.textContent = s.state; }
    if (video && s.videoUrl) { video.src = s.videoUrl; video.hidden = false; }
  }

  function follow(storyId) {
    es?.close();
    es = new EventSource(`/api/story/${storyId}/events`);
    es.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === 'snapshot') {
        (m.story?.segments || []).forEach(applyShot);
        if (m.story?.stage) setStatus(`Rendering — ${m.story.stage}`);
        return;
      }
      if (m.type === 'segment') { applyShot(m); if (m.stage) setStatus(`Rendering — ${m.stage}`); return; }
      if (m.type === 'stage') { setStatus(m.stage === 'stitching' ? 'Stitching the shots…' : m.stage); return; }
      if (m.type === 'done') {
        setStatus(`Done — ${Math.round(m.durationSec || 0)}s in ${m.elapsedSec}s (${m.method}).`, 'ok');
        el.finalBox.hidden = false;
        el.video.src = m.videoUrl;
        el.download.href = m.videoUrl;
        el.download.setAttribute('download', `${(board.title || 'storyboard').replace(/[^\w-]+/g, '-')}.mp4`);
        el.finalNote.textContent =
          `${board.shots.length} shots · ${Math.round(m.durationSec || 0)}s · joined by ${m.method}`;
        done();
        return;
      }
      if (m.type === 'error') { setStatus(m.message, 'error'); done(); }
    };
    es.onerror = () => { es.close(); };
  }

  function done() {
    es?.close();
    es = null;
    el.renderBtn.disabled = false;
    el.planBtn.disabled = false;
    el.framesBtn.disabled = false;
  }
}
