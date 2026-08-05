/* ==========================================================================
   H3 Studio — front end for a self-hosted MiniMax H3 (video + native audio)
   running in ComfyUI on a remote RunPod RTX PRO 6000.

   No build step, no dependencies, no network requests off-origin.
   Everything talks to the Express backend under /api.
   ========================================================================== */

/* ───────────────────────── tiny helpers ───────────────────────── */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** fetch + JSON with a uniform error shape: throws Error with .status and the
 *  server's own `error` string as the message whenever there is one. */
async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, { credentials: 'same-origin', ...opts });
  } catch (netErr) {
    const e = new Error('Could not reach the server. Is the web app still running?');
    e.status = 0;
    e.cause = netErr;
    throw e;
  }
  const text = await res.text();
  let body = null;
  if (text) { try { body = JSON.parse(text); } catch { /* non-JSON body */ } }
  if (!res.ok) {
    const e = new Error((body && (body.error || body.message)) || `HTTP ${res.status}`);
    e.status = res.status;
    e.body = body;
    throw e;
  }
  return body;
}

function fmtDuration(sec) {
  if (sec == null || !isFinite(sec)) return '–';
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m) return `${m}m ${String(s).padStart(2, '0')}s`;
  return `${s}s`;
}
function fmtClock(sec) {                    // mm:ss, for ETAs
  if (sec == null || !isFinite(sec) || sec < 0) return '–';
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}
function fmtBytesMb(mb) {
  if (mb == null) return '–';
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}
function truncate(s, n) { s = s || ''; return s.length > n ? s.slice(0, n - 1) + '…' : s; }
function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ───────────────────────── toasts ───────────────────────── */

const toastHost = $('#toasts');
function toast(message, kind = 'info', ms = 5200) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.dataset.kind = kind;
  el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  el.textContent = message;
  toastHost.appendChild(el);
  const kill = () => {
    el.classList.add('is-out');
    el.addEventListener('animationend', () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 400);
  };
  const t = setTimeout(kill, ms);
  el.addEventListener('click', () => { clearTimeout(t); kill(); });
  return el;
}

/* ───────────────────────── theme ───────────────────────── */

const THEME_KEY = 'h3.theme';
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $('#theme-toggle');
  if (btn) btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
}
(function initTheme() {
  // Run before anything else so the flash is at most one frame.
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === 'dark' || stored === 'light') { applyTheme(stored); return; }
  applyTheme(window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
})();
window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', ev => {
  if (!localStorage.getItem(THEME_KEY)) applyTheme(ev.matches ? 'light' : 'dark');
});
$('#theme-toggle').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
});

/* ───────────────────────── frame-grid maths ─────────────────────────
   The MiniMax H3 video VAE is f16t4d24 with 1x2x2 patchification, so ComfyUI
   declares `length` as min=5, step=17 — i.e. only frame counts satisfying
   `length % 17 == 5` are legal: 5, 22, 39, 56, ... At 24 fps that turns into a
   coarse duration grid (0.21s, 0.92s, 1.63s, 2.33s, 3.04s ...). We reproduce the
   snap client-side purely so the slider label can be live and honest; the server
   re-snaps authoritatively and returns `actualSeconds` + `length`.
------------------------------------------------------------------------ */

const FPS = 24;
const GRID_STEP = 17;       // length % 17 == 5
const GRID_BASE = 5;
const GRID_MAX_N = 20;      // 5 + 20*17 = 345 frames = 14.375 s (model caps at 15 s)
const VALIDATED_MAX_FRAMES = 243; // longest length actually measured on this pod

/** nearest legal frame count for a wall-clock duration in seconds */
function snapFrames(seconds) {
  const rawFrames = seconds * FPS;
  const n = clamp(Math.round((rawFrames - GRID_BASE) / GRID_STEP), 0, GRID_MAX_N);
  return GRID_BASE + n * GRID_STEP;
}
const framesToSeconds = frames => frames / FPS;

/* ───────────────────────── time + cost estimate ─────────────────────────
   Calibrated against three measured runs on this exact card (RTX PRO 6000
   Blackwell, 20 steps unless noted):
       124 frames @ 1344x768        -> 370 s wall clock  (17.4 s/step)
       243 frames @ 1344x768        -> 917 s wall clock  (44.4 s/step)
        39 frames @  640x352, 10 st ->  ~30 s of sampling ( ~3 s/step)
   A pure power law in (pixels x frames) cannot fit all three: the small run is
   dominated by fixed per-step overhead. So the model is
       perStep = BASE + K * U^1.5      with U = width*height*frames / 1e6
   which reproduces 17.4 and 44.4 exactly and lands within ~30% of the small run,
   plus a flat ~25 s for VAE decode + mux. It is a ballpark, and the UI says so.
------------------------------------------------------------------------ */

const EST = { base: 1.9, k: 0.0107, exp: 1.5, overhead: 25 };
const GPU_USD_PER_HOUR = 2.09;

function estimateSeconds(width, height, frames, steps) {
  const u = (width * height * frames) / 1e6;
  const perStep = EST.base + EST.k * Math.pow(u, EST.exp);
  return steps * perStep + EST.overhead;
}

/* ───────────────────────── prompt content ───────────────────────── */

const TEMPLATE = `<look, lens, lighting, setting — one paragraph>

Timeline:
[0s-2s] <shot, camera move, action>
[2s-5s] <... and says clearly in English: "the actual spoken line">
[5s-8s] <resolution beat>

<camera grammar: hard cuts or one continuous shot, no zooms>

Audio: <ambience and its stereo placement>, <voice and its exact time window>, <music/sfx entrances by timestamp>.

No on-screen text, no subtitles.`;

const EXAMPLES = [
  {
    label: 'Ramen stall, Tokyo rain',
    text: `A cramped ramen stall under a dripping awning in a neon back alley, shot on a 35mm lens at f/2, shallow depth of field, wet asphalt throwing magenta and cyan reflections, steam curling through the light.

Timeline:
[0s-2s] Slow push in on the chef ladling broth, his face lit from below by the counter lamp.
[2s-5s] He looks up at the camera, wipes his hands on his apron and says clearly in English: "You picked the right night for this."
[5s-8s] He slides the bowl across the counter; steam blooms into the lens and the neon sign flickers once.

One continuous handheld shot, no cuts, no zooms.

Audio: steady rain and alley reverb spread wide across the stereo field, a low fryer hiss centred, his warm mid-range voice centred from 2.2s to 4.6s, a scooter passing left to right at 6s.

No on-screen text, no subtitles.`
  },
  {
    label: 'Orbital greenhouse',
    text: `The greenhouse module of a low-orbit station, 28mm lens, hard white sunlight raking through a porthole and cutting through drifting water droplets, rows of lettuce trembling in the airflow, everything faintly desaturated except the green.

Timeline:
[0s-2s] Locked-off wide of the module; a botanist floats into frame from the left, catching a rail to stop herself.
[2s-5s] She turns to the camera, holds up a seedling and says clearly in English: "Third generation. None of these have ever felt gravity."
[5s-8s] She releases the seedling; it rotates slowly in front of the porthole as Earth slides past behind it.

One continuous shot, slow drift with the subject, no zooms.

Audio: constant low ventilation hum filling both channels, pumps ticking softly to the right, her close-mic'd voice centred from 2.4s to 5.0s with the room's metallic reflections, a single soft chime at 6.5s.

No on-screen text, no subtitles.`
  },
  {
    label: 'Lighthouse in a gale',
    text: `The lamp room of a north-Atlantic lighthouse at 3am, 50mm lens, the rotating beam sweeping the frame every two seconds and blowing out the highlights, rain lashing the glass, the keeper's face alternating between amber light and near-black.

Timeline:
[0s-2s] The beam sweeps across the keeper as he braces a shoulder against the door.
[2s-5s] He turns into the light and says clearly in English: "She has been out there for six hours. We go now or we do not go."
[5s-8s] Hard cut to the exterior: the beam cutting into horizontal rain, the tower shrinking against black water.

One hard cut at 5s. No zooms, no camera shake beyond the wind.

Audio: gale and rain wide and unstable across the stereo image, the lamp motor grinding low and centred with a rhythmic sweep every 2s, his shouted voice centred from 2.3s to 4.8s, a foghorn entering far right at 6s.

No on-screen text, no subtitles.`
  },
  {
    label: 'Workshop, close macro',
    text: `A cluttered watchmaker's bench in a shuttered shop, 100mm macro lens, single warm desk lamp, dust suspended in the beam, brass and steel shavings catching the light against deep brown shadow.

Timeline:
[0s-2s] Extreme close on tweezers lowering a balance wheel into a movement; only the hands are in frame.
[2s-5s] The camera lifts to reveal an old woman behind a loupe; she says clearly in English: "Everything in here wants to stop. My job is to argue with it."
[5s-8s] She wipes the crystal with her sleeve and the second hand begins to move.

One continuous slow crane up, no cuts, no zooms.

Audio: a dead-quiet room tone, the ticking of a dozen clocks scattered unevenly across the stereo field, tiny metallic clinks centred from 0s to 2s, her dry close voice centred from 2.6s to 5.2s, one clean ticking movement entering centre at 7s.

No on-screen text, no subtitles.`
  }
];

/* ───────────────────────── application state ───────────────────────── */

const SETTINGS_KEY = 'h3.settings';

const state = {
  authed: false,
  noAuth: false,
  mode: 't2v',
  status: null,
  deadlineAt: null,     // ms epoch, derived from status.deadline.secondsLeft
  assets: { firstFrame: [], lastFrame: [], refs: [] },
  submitting: false,    // POST /api/generate in flight
  job: null,            // { id, params, stage }
  current: null,        // job object currently shown in the result panel
  lastSeed: null,
  jobs: []
};

/* ───────────────────────── element refs ───────────────────────── */

const el = {
  boot: $('#boot'),
  login: $('#login'),
  loginForm: $('#login-form'),
  loginPassword: $('#login-password'),
  loginError: $('#login-error'),
  loginSubmit: $('#login-submit'),
  app: $('#app'),

  dot: $('#status-dot'),
  statusText: $('#status-text'),
  statusLive: $('#status-live'),
  gpuName: $('#gpu-name'),
  vramFill: $('#vram-fill'),
  vramText: $('#vram-text'),
  queueText: $('#queue-text'),
  statDeadline: $('#stat-deadline'),
  deadlineText: $('#deadline-text'),
  offlineBanner: $('#offline-banner'),
  offlineDetail: $('#offline-detail'),

  tabs: $$('.tab'),
  panes: { t2v: $('#pane-t2v'), i2v: $('#pane-i2v'), flf2v: $('#pane-flf2v'), r2v: $('#pane-r2v') },
  refCounts: $('#ref-counts'),
  r2vNote: $('#r2v-note'),
  refImageSize: $('#ref-image-size'),

  prompt: $('#prompt'),
  promptCount: $('#prompt-count'),
  insertTemplate: $('#insert-template'),
  toggleStructure: $('#toggle-structure'),
  structurePanel: $('#structure-panel'),
  clearPrompt: $('#clear-prompt'),
  exampleChips: $('#example-chips'),

  preset: $('#preset'),
  advRes: $('#adv-res'),
  width: $('#width'),
  height: $('#height'),
  duration: $('#duration'),
  durationOut: $('#duration-out'),
  durWarn: $('#dur-warn'),
  steps: $('#steps'),
  stepsOut: $('#steps-out'),
  seed: $('#seed'),
  seedRandom: $('#seed-random'),
  seedReuse: $('#seed-reuse'),

  estTime: $('#est-time'),
  estCost: $('#est-cost'),
  estDetail: $('#est-detail'),

  generate: $('#generate'),
  generateLabel: $('#generate-label'),
  generateReason: $('#generate-reason'),

  progressCard: $('#progress-card'),
  progressStage: $('#progress-stage'),
  progressTrack: $('#progress-bar-wrap'),
  progressFill: $('#progress-fill'),
  progressSteps: $('#progress-steps'),
  progressEta: $('#progress-eta'),
  cancel: $('#cancel'),

  result: $('#result'),
  video: $('#video'),
  unmute: $('#unmute'),
  resultMeta: $('#result-meta'),
  resultPrompt: $('#result-prompt'),
  download: $('#download'),
  copySettings: $('#copy-settings'),
  regen: $('#regen'),

  placeholder: $('#placeholder'),
  errorbox: $('#errorbox'),
  errorboxMsg: $('#errorbox-msg'),
  errorboxDismiss: $('#errorbox-dismiss'),

  gallery: $('#gallery'),
  galleryEmpty: $('#gallery-empty'),
  refreshGallery: $('#refresh-gallery'),

  fileInput: $('#file-input')
};

/* ───────────────────────── session / login ───────────────────────── */

async function boot() {
  let session;
  try {
    session = await api('/api/session');
  } catch (err) {
    el.boot.querySelector('.boot-inner').innerHTML =
      `<span style="color:var(--danger)">Cannot reach the web app: ${escapeHtml(err.message)}</span>`;
    return;
  }
  state.authed = !!session.authed;
  state.noAuth = !!session.noAuth;
  el.boot.hidden = true;

  if (state.authed || state.noAuth) enterApp();
  else { el.login.hidden = false; el.loginPassword.focus(); }
}

el.loginForm.addEventListener('submit', async ev => {
  ev.preventDefault();
  el.loginError.hidden = true;
  el.loginSubmit.disabled = true;
  el.loginSubmit.textContent = 'Checking…';
  try {
    const r = await api('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: el.loginPassword.value })
    });
    state.authed = true;
    state.noAuth = !!r.noAuth;
    el.login.hidden = true;
    enterApp();
  } catch (err) {
    el.loginError.textContent = err.status === 401
      ? (err.message || 'That password was not accepted.')
      : `Login failed: ${err.message}`;
    el.loginError.hidden = false;
    el.loginPassword.select();
  } finally {
    el.loginSubmit.disabled = false;
    el.loginSubmit.textContent = 'Unlock';
  }
});

function enterApp() {
  el.app.hidden = false;
  buildExamples();
  restoreSettings();
  syncAll();
  pollStatus();
  setInterval(pollStatus, 10_000);
  setInterval(tickDeadline, 1000);
  loadGallery();
}

/* ───────────────────────── status polling ───────────────────────── */

let lastOnline = null;

async function pollStatus() {
  let s;
  try {
    s = await api('/api/status');
  } catch (err) {
    s = { online: false, error: err.message };
  }
  state.status = s;
  renderStatus(s);
  updateGenerateAvailability();
}

function renderStatus(s) {
  const online = !!s.online;

  el.dot.dataset.state = online ? 'online' : 'offline';
  el.statusText.textContent = online ? 'Online' : 'Offline';
  el.statusText.style.color = online ? 'var(--ok)' : 'var(--danger)';

  const gpu = s.gpu || {};
  el.gpuName.textContent = gpu.name || 'GPU';
  el.gpuName.title = gpu.name || '';
  const used = gpu.vramUsedMb, total = gpu.vramTotalMb;
  if (used != null && total) {
    el.vramFill.style.width = `${clamp((used / total) * 100, 0, 100).toFixed(1)}%`;
    el.vramText.textContent = `${fmtBytesMb(used)} / ${fmtBytesMb(total)}`;
    el.vramText.title = gpu.utilPct != null ? `GPU utilisation ${gpu.utilPct}%` : '';
  } else {
    el.vramFill.style.width = '0%';
    el.vramText.textContent = '– / –';
  }

  const q = s.queue || {};
  const running = q.running ?? 0, pending = q.pending ?? 0;
  el.queueText.textContent = `${running} / ${pending}`;
  el.queueText.title = `${running} running, ${pending} pending`;

  // deadline: keep an absolute target so the countdown ticks smoothly between polls
  const dl = s.deadline || {};
  if (dl.secondsLeft != null) {
    state.deadlineAt = Date.now() + dl.secondsLeft * 1000;
    el.statDeadline.hidden = false;
    el.statDeadline.title = dl.iso
      ? `Pod auto-terminates at ${new Date(dl.iso).toLocaleString()} — a deliberate cost cap`
      : 'The pod is killed on a timer to cap cost';
  } else if (dl.iso) {
    state.deadlineAt = new Date(dl.iso).getTime();
    el.statDeadline.hidden = false;
  } else {
    state.deadlineAt = null;
    el.statDeadline.hidden = true;
  }
  tickDeadline();

  // offline banner
  el.offlineBanner.hidden = online;
  if (!online) {
    el.offlineDetail.textContent = s.error
      ? `${s.error} — generation is unavailable until the pod is back. Clips already rendered still play.`
      : 'The RunPod instance is gone or unreachable, so generation is unavailable. Clips already rendered still play.';
  }

  // model availability -> r2v tab
  const models = s.models || {};
  const r2vTab = el.tabs.find(t => t.dataset.mode === 'r2v');
  const r2vOk = !!models.ref2va;
  r2vTab.setAttribute('aria-disabled', String(!r2vOk));
  r2vTab.title = r2vOk
    ? 'Drive a new clip from reference images, videos and voices'
    : 'Unavailable: the Ref2VA checkpoint is not loaded on the pod. Only the fl2va checkpoint is present.';
  el.r2vNote.hidden = r2vOk;
  if (!r2vOk && state.mode === 'r2v') setMode('t2v');

  if (online && models.fl2va === false) {
    el.offlineBanner.hidden = false;
    el.offlineDetail.textContent =
      'ComfyUI is up but the fl2va checkpoint is not loaded, so generation will fail. Check the model symlinks on the pod.';
  }

  // aria-live only on transitions, so screen readers are not spammed every 10s
  if (lastOnline !== online) {
    el.statusLive.textContent = online
      ? `Pod online. ${gpu.name || 'GPU'} ready.`
      : `Pod offline. ${s.error || 'Generation unavailable.'}`;
    lastOnline = online;
  }
}

function tickDeadline() {
  if (state.deadlineAt == null) return;
  const left = Math.max(0, Math.round((state.deadlineAt - Date.now()) / 1000));
  const h = Math.floor(left / 3600), m = Math.floor((left % 3600) / 60), s = left % 60;
  el.deadlineText.textContent = left === 0
    ? 'expired'
    : (h ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m ${String(s).padStart(2, '0')}s`);
  const mins = left / 60;
  el.statDeadline.dataset.level = mins <= 10 ? 'danger' : mins <= 30 ? 'warn' : 'ok';
}

/* ───────────────────────── mode tabs ───────────────────────── */

function setMode(mode) {
  state.mode = mode;
  el.tabs.forEach(t => {
    const on = t.dataset.mode === mode;
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
  });
  Object.entries(el.panes).forEach(([m, pane]) => { pane.hidden = m !== mode; });
  updateGenerateAvailability();
  saveSettings();
}

el.tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.getAttribute('aria-disabled') === 'true') {
      toast(tab.title || 'That mode is unavailable right now.', 'warn');
      return;
    }
    setMode(tab.dataset.mode);
  });
  // roving tabindex: arrow keys move between tabs, as a tablist should
  tab.addEventListener('keydown', ev => {
    const i = el.tabs.indexOf(tab);
    let next = null;
    if (ev.key === 'ArrowRight') next = el.tabs[(i + 1) % el.tabs.length];
    if (ev.key === 'ArrowLeft') next = el.tabs[(i - 1 + el.tabs.length) % el.tabs.length];
    if (ev.key === 'Home') next = el.tabs[0];
    if (ev.key === 'End') next = el.tabs[el.tabs.length - 1];
    if (next) { ev.preventDefault(); next.focus(); next.click(); }
  });
});

/* ───────────────────────── uploads + drop zones ─────────────────────────
   One hidden <input type=file> is shared by every zone; `pendingZone` records
   which zone opened it. Every zone supports click, drag-drop and paste.
------------------------------------------------------------------------ */

const REF_LIMITS = { image: 9, video: 3, audio: 3, total: 12 };
let pendingZone = null;
let uid = 0;

function kindOfFile(file) {
  const t = (file.type || '').toLowerCase();
  if (t.startsWith('image/')) return 'image';
  if (t.startsWith('video/')) return 'video';
  if (t.startsWith('audio/')) return 'audio';
  // fall back to the extension when the browser gives us nothing
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  if (['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'avif'].includes(ext)) return 'image';
  if (['mp4', 'mov', 'webm', 'mkv', 'avi', 'm4v'].includes(ext)) return 'video';
  if (['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'].includes(ext)) return 'audio';
  return null;
}

function slotOf(zone) { return zone.dataset.slot; }
function assetsFor(zone) { return state.assets[slotOf(zone)]; }

function acceptAttrFor(zone) {
  const kinds = zone.dataset.accept.split(',');
  return kinds.map(k => `${k}/*`).join(',');
}

/** Validate a batch against the zone's limits. Returns {ok, rejected:[{file,why}]} */
function filterForZone(zone, files) {
  const accepted = zone.dataset.accept.split(',');
  const max = parseInt(zone.dataset.max, 10);
  const existing = assetsFor(zone);
  const ok = [];
  const rejected = [];

  // count what we already hold, per kind (only meaningful for the ref zone)
  const counts = { image: 0, video: 0, audio: 0 };
  existing.forEach(a => { if (counts[a.kind] != null) counts[a.kind]++; });

  for (const file of files) {
    const kind = kindOfFile(file);
    if (!kind || !accepted.includes(kind)) {
      rejected.push({ file, why: `${file.name}: this zone only takes ${accepted.join(', ')} files.` });
      continue;
    }
    if (existing.length + ok.length >= max) {
      rejected.push({ file, why: `${file.name}: ${max === 1 ? 'this slot holds one file' : `at most ${max} files total`}.` });
      continue;
    }
    if (max > 1 && REF_LIMITS[kind] != null && counts[kind] >= REF_LIMITS[kind]) {
      rejected.push({ file, why: `${file.name}: at most ${REF_LIMITS[kind]} ${kind} files.` });
      continue;
    }
    counts[kind]++;
    ok.push({ file, kind });
  }
  return { ok, rejected };
}

function addFiles(zone, fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;

  // single-slot zones replace rather than reject
  if (parseInt(zone.dataset.max, 10) === 1 && assetsFor(zone).length) {
    assetsFor(zone).length = 0;
  }

  const { ok, rejected } = filterForZone(zone, files);
  rejected.forEach(r => toast(r.why, 'warn', 6000));
  ok.forEach(({ file, kind }) => uploadFile(zone, file, kind));
}

function uploadFile(zone, file, kind) {
  const asset = {
    id: `a${++uid}`,
    name: null,
    kind,
    fileName: file.name,
    previewUrl: null,
    localUrl: (kind === 'image' || kind === 'video') ? URL.createObjectURL(file) : null,
    status: 'uploading',
    pct: 0
  };
  assetsFor(zone).push(asset);
  renderZone(zone);

  const form = new FormData();
  form.append('file', file, file.name);

  // XHR rather than fetch: we want a real upload progress bar
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload');
  xhr.withCredentials = true;
  xhr.upload.addEventListener('progress', ev => {
    if (!ev.lengthComputable) return;
    asset.pct = (ev.loaded / ev.total) * 100;
    const bar = zone.querySelector(`[data-asset="${asset.id}"] .thumb-progress > span`);
    if (bar) bar.style.width = `${asset.pct}%`;
  });
  xhr.addEventListener('load', () => {
    let body = null;
    try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
    if (xhr.status >= 200 && xhr.status < 300 && body && body.name) {
      asset.name = body.name;
      asset.kind = body.kind || asset.kind;
      asset.previewUrl = body.previewUrl || `/api/asset/${encodeURIComponent(body.name)}`;
      asset.status = 'done';
    } else {
      asset.status = 'error';
      asset.error = (body && body.error) || `Upload failed (HTTP ${xhr.status})`;
      toast(`${file.name}: ${asset.error}`, 'error', 7000);
    }
    renderZone(zone);
    updateGenerateAvailability();
  });
  xhr.addEventListener('error', () => {
    asset.status = 'error';
    asset.error = 'Network error during upload';
    toast(`${file.name}: upload could not reach the server.`, 'error');
    renderZone(zone);
  });
  xhr.addEventListener('abort', () => {
    const arr = assetsFor(zone);
    const i = arr.indexOf(asset);
    if (i >= 0) arr.splice(i, 1);
    renderZone(zone);
  });
  asset.xhr = xhr;
  xhr.send(form);
}

function renderZone(zone) {
  const items = zone.querySelector('.dz-items');
  const assets = assetsFor(zone);
  zone.classList.toggle('has-items', assets.length > 0);
  items.innerHTML = '';

  assets.forEach(a => {
    const wrap = document.createElement('div');
    wrap.className = 'thumb';
    wrap.dataset.asset = a.id;
    if (a.status === 'uploading') wrap.classList.add('is-uploading');
    if (a.status === 'error') wrap.classList.add('is-error');
    wrap.title = `${a.fileName}${a.error ? ' — ' + a.error : ''}`;

    const src = a.localUrl || a.previewUrl;
    if (a.kind === 'image' && src) {
      const img = document.createElement('img');
      img.src = src; img.alt = a.fileName;
      wrap.appendChild(img);
    } else if (a.kind === 'video' && src) {
      const v = document.createElement('video');
      v.src = src; v.muted = true; v.playsInline = true; v.preload = 'metadata';
      wrap.appendChild(v);
    } else {
      const k = document.createElement('span');
      k.className = 'thumb-kind';
      k.textContent = a.kind === 'audio' ? '♪ audio' : truncate(a.fileName, 16);
      wrap.appendChild(k);
    }

    if (a.status === 'uploading') {
      const p = document.createElement('div');
      p.className = 'thumb-progress';
      const bar = document.createElement('span');
      bar.style.width = `${a.pct}%`;
      p.appendChild(bar);
      wrap.appendChild(p);
    }

    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'thumb-remove';
    rm.textContent = '×';
    rm.setAttribute('aria-label', `Remove ${a.fileName}`);
    rm.addEventListener('click', ev => {
      ev.stopPropagation();
      if (a.status === 'uploading' && a.xhr) { a.xhr.abort(); return; }
      const i = assets.indexOf(a);
      if (i >= 0) assets.splice(i, 1);
      if (a.localUrl) URL.revokeObjectURL(a.localUrl);
      renderZone(zone);
      updateGenerateAvailability();
    });
    wrap.appendChild(rm);
    items.appendChild(wrap);
  });

  // browse affordance once the zone has content and the dz-body button is hidden
  const max = parseInt(zone.dataset.max, 10);
  if (assets.length && assets.length < max) {
    const add = document.createElement('button');
    add.type = 'button';
    add.className = 'thumb thumb-add';
    add.innerHTML = '<span class="thumb-kind">+ add</span>';
    add.setAttribute('aria-label', 'Add more files');
    add.addEventListener('click', ev => { ev.stopPropagation(); openPicker(zone); });
    items.appendChild(add);
  } else if (assets.length && max === 1) {
    const rep = document.createElement('button');
    rep.type = 'button';
    rep.className = 'thumb thumb-add';
    rep.innerHTML = '<span class="thumb-kind">replace</span>';
    rep.setAttribute('aria-label', 'Replace this file');
    rep.addEventListener('click', ev => { ev.stopPropagation(); openPicker(zone); });
    items.appendChild(rep);
  }

  if (zone.dataset.slot === 'refs') updateRefCounts();
}

function updateRefCounts() {
  const c = { image: 0, video: 0, audio: 0 };
  state.assets.refs.forEach(a => { if (c[a.kind] != null) c[a.kind]++; });
  el.refCounts.textContent =
    `${c.image}/9 images · ${c.video}/3 videos · ${c.audio}/3 audio · ${state.assets.refs.length}/12 total`;
}

function openPicker(zone) {
  pendingZone = zone;
  el.fileInput.accept = acceptAttrFor(zone);
  el.fileInput.multiple = parseInt(zone.dataset.max, 10) > 1;
  el.fileInput.value = '';
  el.fileInput.click();
}

function wireZone(zone) {
  // clicking dead space in the zone also opens the picker, but never swallow a
  // click that was meant for one of the buttons inside it
  zone.addEventListener('click', ev => {
    if (ev.target.closest('button')) return;
    openPicker(zone);
  });
  zone.querySelector('.dz-body').addEventListener('click', () => openPicker(zone));
  ['dragenter', 'dragover'].forEach(evName =>
    zone.addEventListener(evName, ev => { ev.preventDefault(); zone.classList.add('is-drag'); }));
  ['dragleave', 'drop'].forEach(evName =>
    zone.addEventListener(evName, ev => { ev.preventDefault(); zone.classList.remove('is-drag'); }));
  zone.addEventListener('drop', ev => {
    ev.preventDefault();
    addFiles(zone, ev.dataTransfer && ev.dataTransfer.files);
  });
}

$$('.dz').forEach(wireZone);

el.fileInput.addEventListener('change', () => {
  if (pendingZone) addFiles(pendingZone, el.fileInput.files);
  pendingZone = null;
});

// paste: route to the focused zone, else the first zone of the active mode
document.addEventListener('paste', ev => {
  if (el.app.hidden) return;
  const files = Array.from((ev.clipboardData && ev.clipboardData.files) || []);
  if (!files.length) return;
  const focused = document.activeElement && document.activeElement.closest && document.activeElement.closest('.dz');
  const zone = focused || el.panes[state.mode].querySelector('.dz');
  if (!zone) {
    toast('This mode has no upload slot — switch to Image → Video or Reference → Video to paste files.', 'warn');
    return;
  }
  ev.preventDefault();
  addFiles(zone, files);
  toast(`Pasted ${files.length} file${files.length > 1 ? 's' : ''}.`, 'ok', 2600);
});

/* ───────────────────────── prompt ───────────────────────── */

function updatePromptCount() {
  const n = el.prompt.value.length;
  el.promptCount.textContent = `${n.toLocaleString()} character${n === 1 ? '' : 's'}`;
}
el.prompt.addEventListener('input', () => { updatePromptCount(); updateGenerateAvailability(); saveSettings(); });

el.insertTemplate.addEventListener('click', () => {
  if (el.prompt.value.trim() && !confirm('Replace the current prompt with the template?')) return;
  el.prompt.value = TEMPLATE;
  el.prompt.focus();
  el.prompt.setSelectionRange(0, 0);
  el.prompt.scrollTop = 0;
  updatePromptCount();
  updateGenerateAvailability();
  saveSettings();
});

el.clearPrompt.addEventListener('click', () => {
  el.prompt.value = '';
  updatePromptCount();
  updateGenerateAvailability();
  saveSettings();
  el.prompt.focus();
});

el.toggleStructure.addEventListener('click', () => {
  const open = el.structurePanel.hidden;
  el.structurePanel.hidden = !open;
  el.toggleStructure.setAttribute('aria-expanded', String(open));
  el.toggleStructure.textContent = open ? 'Prompt structure ▴' : 'Prompt structure ▾';
});

function buildExamples() {
  el.exampleChips.innerHTML = '';
  EXAMPLES.forEach(ex => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = ex.label;
    b.title = truncate(ex.text.split('\n')[0], 140);
    b.addEventListener('click', () => {
      el.prompt.value = ex.text;
      updatePromptCount();
      updateGenerateAvailability();
      saveSettings();
      el.prompt.focus();
      el.prompt.setSelectionRange(0, 0);
      el.prompt.scrollTop = 0;
    });
    el.exampleChips.appendChild(b);
  });
}

/* ───────────────────────── settings ───────────────────────── */

function currentDims() {
  return {
    width: clamp(parseInt(el.width.value, 10) || 1344, 256, 2048),
    height: clamp(parseInt(el.height.value, 10) || 768, 256, 2048)
  };
}

function syncPresetFromDims() {
  const { width, height } = currentDims();
  const key = `${width}x${height}`;
  const has = Array.from(el.preset.options).some(o => o.value === key);
  el.preset.value = has ? key : 'custom';
}

el.preset.addEventListener('change', () => {
  if (el.preset.value === 'custom') { el.advRes.open = true; el.width.focus(); return; }
  const [w, h] = el.preset.value.split('x').map(Number);
  el.width.value = w;
  el.height.value = h;
  syncAll();
});

[el.width, el.height].forEach(inp => inp.addEventListener('input', () => {
  syncPresetFromDims();
  syncEstimate();
  saveSettings();
}));
[el.width, el.height].forEach(inp => inp.addEventListener('change', () => {
  // snap to the 32px grid the model wants
  const v = parseInt(inp.value, 10);
  if (isFinite(v)) inp.value = clamp(Math.round(v / 32) * 32, 256, 2048);
  syncPresetFromDims();
  syncEstimate();
  saveSettings();
}));

el.duration.addEventListener('input', () => { syncDuration(); syncEstimate(); });
el.duration.addEventListener('change', saveSettings);

function syncDuration() {
  const requested = parseFloat(el.duration.value);
  const frames = snapFrames(requested);
  const secs = framesToSeconds(frames);
  el.durationOut.textContent = `${secs.toFixed(2)} s · ${frames} frames`;
  el.duration.title = `Requested ${requested.toFixed(2)} s → snapped to ${frames} frames (${secs.toFixed(2)} s)`;
  el.duration.setAttribute('aria-valuetext', `${secs.toFixed(2)} seconds, ${frames} frames`);
  el.durWarn.hidden = frames <= VALIDATED_MAX_FRAMES;
  return { frames, secs };
}

el.steps.addEventListener('input', () => { el.stepsOut.textContent = el.steps.value; syncEstimate(); });
el.steps.addEventListener('change', saveSettings);

el.seedRandom.addEventListener('click', () => {
  el.seed.value = Math.floor(Math.random() * 4294967295);
  saveSettings();
});
el.seed.addEventListener('input', saveSettings);
el.seedReuse.addEventListener('click', () => {
  if (state.lastSeed == null) return;
  el.seed.value = state.lastSeed;
  saveSettings();
  toast(`Seed ${state.lastSeed} restored.`, 'ok', 2600);
});

function syncEstimate() {
  const { width, height } = currentDims();
  const { frames } = syncDurationSilently();
  const steps = parseInt(el.steps.value, 10);
  const secs = estimateSeconds(width, height, frames, steps);
  const cost = (secs / 3600) * GPU_USD_PER_HOUR;

  el.estTime.textContent = secs < 90 ? `≈ ${Math.round(secs)} s` : `≈ ${(secs / 60).toFixed(1)} min`;
  el.estCost.textContent = `≈ $${cost.toFixed(2)}`;
  el.estDetail.textContent = `${width}×${height} · ${frames} frames · ${steps} steps`;
}
// duration label is owned by syncDuration(); this variant only needs the number
function syncDurationSilently() {
  const frames = snapFrames(parseFloat(el.duration.value));
  return { frames, secs: framesToSeconds(frames) };
}

function syncAll() {
  syncPresetFromDims();
  syncDuration();
  el.stepsOut.textContent = el.steps.value;
  updatePromptCount();
  syncEstimate();
  updateRefCounts();
  updateGenerateAvailability();
  el.seedReuse.disabled = state.lastSeed == null;
}

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({
      mode: state.mode,
      prompt: el.prompt.value,
      width: el.width.value,
      height: el.height.value,
      duration: el.duration.value,
      steps: el.steps.value,
      seed: el.seed.value,
      refImageSize: el.refImageSize.value,
      lastSeed: state.lastSeed
    }));
  } catch { /* private mode / quota — settings are a nicety, not a requirement */ }
}

function restoreSettings() {
  let s = null;
  try { s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || 'null'); } catch { /* ignore */ }
  if (!s) { setMode('t2v'); return; }
  if (s.prompt) el.prompt.value = s.prompt;
  if (s.width) el.width.value = s.width;
  if (s.height) el.height.value = s.height;
  if (s.duration) el.duration.value = s.duration;
  if (s.steps) el.steps.value = s.steps;
  if (s.seed) el.seed.value = s.seed;
  if (s.refImageSize) el.refImageSize.value = s.refImageSize;
  if (s.lastSeed != null) state.lastSeed = s.lastSeed;
  setMode(['t2v', 'i2v', 'flf2v', 'r2v'].includes(s.mode) ? s.mode : 't2v');
}
el.refImageSize.addEventListener('change', saveSettings);

/* ───────────────────────── generate availability ───────────────────────── */

function readySlot(slot) { return state.assets[slot].some(a => a.status === 'done'); }

function blockedReason() {
  if (state.job || state.submitting) return 'A job is already running — cancel it or wait for it to finish.';
  if (state.status && !state.status.online) return 'The pod is offline, so nothing can be queued.';
  if (state.status && state.status.models && state.status.models.fl2va === false)
    return 'The fl2va checkpoint is not loaded on the pod.';
  if (!el.prompt.value.trim()) return 'Write a prompt first.';
  const uploading = Object.values(state.assets).flat().some(a => a.status === 'uploading');
  if (uploading) return 'Waiting for uploads to finish…';
  if (state.mode === 'i2v' && !readySlot('firstFrame')) return 'Image → Video needs a first frame.';
  if (state.mode === 'flf2v' && !readySlot('firstFrame') && !readySlot('lastFrame'))
    return 'Give at least one of the two frames.';
  if (state.mode === 'r2v') {
    if (state.status && state.status.models && !state.status.models.ref2va)
      return 'The Ref2VA checkpoint is not loaded on the pod.';
    if (!state.assets.refs.some(a => a.status === 'done')) return 'Add at least one reference file.';
  }
  return null;
}

function updateGenerateAvailability() {
  const reason = blockedReason();
  el.generate.disabled = !!reason;
  el.generate.setAttribute('aria-disabled', String(!!reason));
  el.generateReason.textContent = reason || '';
  el.generateReason.hidden = !reason;
  el.seedReuse.disabled = state.lastSeed == null;
}

/* ───────────────────────── generation ───────────────────────── */

el.generate.addEventListener('click', () => generate());

async function generate() {
  const reason = blockedReason();
  if (reason) { toast(reason, 'warn'); return; }

  const { width, height } = currentDims();
  const { frames, secs } = syncDurationSilently();
  const steps = parseInt(el.steps.value, 10);
  const seedRaw = el.seed.value.trim();
  const seed = seedRaw === '' ? null : clamp(parseInt(seedRaw, 10) || 0, 0, 4294967295);

  const payload = {
    mode: state.mode,
    prompt: el.prompt.value.trim(),
    width, height,
    durationSec: Number(secs.toFixed(3)),
    steps,
    seed
  };

  if (state.mode === 'i2v' || state.mode === 'flf2v') {
    const ff = state.assets.firstFrame.find(a => a.status === 'done');
    const lf = state.assets.lastFrame.find(a => a.status === 'done');
    if (ff) payload.firstFrame = ff.name;
    if (state.mode === 'flf2v' && lf) payload.lastFrame = lf.name;
  }
  if (state.mode === 'r2v') {
    const done = state.assets.refs.filter(a => a.status === 'done');
    const pick = k => done.filter(a => a.kind === k).map(a => a.name);
    const imgs = pick('image'), vids = pick('video'), auds = pick('audio');
    if (imgs.length) payload.refImages = imgs;
    if (vids.length) payload.refVideos = vids;
    if (auds.length) payload.refAudios = auds;
    payload.refImageSize = el.refImageSize.value;
  }

  hideError();
  state.submitting = true;               // closes the window between click and jobId
  setRunning(true, { stage: 'submitting', totalSteps: steps });

  let job;
  try {
    job = await api('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (err) {
    state.submitting = false;
    setRunning(false);
    if (err.status === 409) {
      showError(`${err.message}\n\nThe pod went away between the status poll and the submit. Wait for the status dot to go green.`);
      pollStatus();
    } else {
      showError(err.message || 'The server refused the job.');
    }
    return;
  }

  // The server snaps duration to the legal frame grid; trust its answer over ours.
  if (job.length && job.length !== frames) {
    toast(`Duration snapped to ${job.length} frames (${(job.actualSeconds ?? job.length / FPS).toFixed(2)} s).`, 'info', 5000);
  }
  state.lastSeed = job.seed ?? seed;
  el.seedReuse.disabled = state.lastSeed == null;
  saveSettings();

  state.job = {
    id: job.jobId,
    params: {
      ...payload,
      steps,
      frames: job.length || frames,
      actualSeconds: job.actualSeconds ?? secs,
      seed: job.seed ?? seed
    },
    startedAt: Date.now()
  };
  state.submitting = false;
  updateGenerateAvailability();
  setProgress({ type: 'queued' });
  subscribe(job.jobId);
}

/* --- progress UI ------------------------------------------------------- */

let etaTimer = null;
let etaTarget = null;   // ms epoch when sampling is predicted to finish

function setRunning(running, init) {
  el.generate.textContent = '';
  const label = document.createElement('span');
  label.id = 'generate-label';
  label.textContent = running ? 'Generating…' : 'Generate video + audio';
  el.generate.appendChild(label);
  el.generateLabel = label;

  el.progressCard.hidden = !running;
  if (running) {
    el.placeholder.hidden = true;
    el.progressStage.textContent = init && init.stage === 'submitting' ? 'Submitting…' : 'Queued';
    el.progressTrack.classList.add('indeterminate');
    el.progressFill.style.width = '0%';
    el.progressSteps.textContent = '';
    el.progressEta.textContent = '';
  } else {
    stopEta();
    if (!state.current) el.placeholder.hidden = false;
  }
  updateGenerateAvailability();
}

function startEta(seconds) {
  etaTarget = Date.now() + seconds * 1000;
  if (etaTimer) return;
  etaTimer = setInterval(() => {
    if (etaTarget == null) return;
    const left = (etaTarget - Date.now()) / 1000;
    el.progressEta.textContent = left > 0 ? `about ${fmtClock(left)} left` : 'finishing up…';
  }, 1000);
}
function stopEta() { clearInterval(etaTimer); etaTimer = null; etaTarget = null; }

function setProgress(ev) {
  switch (ev.type) {
    case 'queued':
      el.progressStage.textContent = ev.queuePosition
        ? `Queued — position ${ev.queuePosition}`
        : 'Queued on the pod';
      el.progressTrack.classList.add('indeterminate');
      el.progressSteps.textContent = 'waiting for a free slot';
      el.progressEta.textContent = '';
      stopEta();
      break;

    case 'start':
      el.progressStage.textContent = 'Sampling';
      el.progressTrack.classList.remove('indeterminate');
      el.progressFill.style.width = '0%';
      el.progressSteps.textContent = 'step 0';
      break;

    case 'progress': {
      const total = ev.totalSteps || (state.job && state.job.params.steps) || 20;
      const pct = ev.pct != null ? ev.pct : (ev.step / total) * 100;
      el.progressStage.textContent = 'Sampling';
      el.progressTrack.classList.remove('indeterminate');
      el.progressFill.style.width = `${clamp(pct, 0, 100)}%`;
      el.progressTrack.setAttribute('aria-valuenow', Math.round(clamp(pct, 0, 100)));
      el.progressSteps.textContent = `step ${ev.step} / ${total}`;
      if (ev.etaSeconds != null) startEta(ev.etaSeconds);
      else el.progressEta.textContent = `${Math.round(clamp(pct, 0, 100))}%`;
      break;
    }

    case 'decoding':
      // No step counter exists for VAE decode + mux; it can run 30-90 s.
      el.progressStage.textContent = 'Decoding video + stereo audio';
      el.progressTrack.classList.add('indeterminate');
      el.progressTrack.removeAttribute('aria-valuenow');
      el.progressSteps.textContent = 'sampling done — VAE decode and mux, no step counter here';
      el.progressEta.textContent = 'usually 30–90 s';
      stopEta();
      break;
  }
}

el.cancel.addEventListener('click', async () => {
  if (!state.job) return;
  const id = state.job.id;
  el.cancel.disabled = true;
  try {
    await api(`/api/cancel/${encodeURIComponent(id)}`, { method: 'POST' });
    toast('Cancellation sent.', 'ok');
  } catch (err) {
    toast(`Could not cancel: ${err.message}`, 'error');
  } finally {
    el.cancel.disabled = false;
  }
  // The stream will normally deliver an error event; if it does not, don't hang.
  setTimeout(() => {
    if (state.job && state.job.id === id) {
      teardownStream();
      state.job = null;
      setRunning(false);
      loadGallery();
    }
  }, 6000);
});

/* --- SSE with a two-stage fallback --------------------------------------
   1. EventSource. If it errors before a terminal event, retry once after 3 s.
   2. If the retry also drops, poll GET /api/jobs/:id every 2.5 s and synthesise
      the same events from the job's `state` field, so the UI code below has one
      shape to deal with either way.
------------------------------------------------------------------------- */

let stream = { es: null, poll: null, retried: false, terminal: false, jobId: null };

function teardownStream() {
  if (stream.es) { try { stream.es.close(); } catch { /* ignore */ } }
  clearInterval(stream.poll);
  stream = { es: null, poll: null, retried: false, terminal: false, jobId: null };
  stopEta();
}

function subscribe(jobId) {
  teardownStream();
  stream.jobId = jobId;

  const open = () => {
    let es;
    try {
      es = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    } catch {
      startPolling(jobId);
      return;
    }
    stream.es = es;

    es.onmessage = msg => {
      let data;
      try { data = JSON.parse(msg.data); } catch { return; }
      handleJobEvent(data);
    };

    es.onerror = () => {
      // EventSource also fires onerror when the server closes the stream after a
      // terminal event — that case is normal and must not trigger a retry.
      if (stream.terminal || stream.jobId !== jobId) return;
      try { es.close(); } catch { /* ignore */ }
      stream.es = null;
      if (!stream.retried) {
        stream.retried = true;
        toast('Progress stream dropped — reconnecting in 3 s…', 'warn', 3000);
        setTimeout(() => { if (stream.jobId === jobId && !stream.terminal) open(); }, 3000);
      } else {
        toast('Progress stream unavailable — falling back to polling.', 'warn', 4000);
        startPolling(jobId);
      }
    };
  };

  open();
}

function startPolling(jobId) {
  clearInterval(stream.poll);
  let lastState = null;
  stream.poll = setInterval(async () => {
    if (stream.jobId !== jobId || stream.terminal) { clearInterval(stream.poll); return; }
    let job;
    try {
      job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    } catch (err) {
      if (err.status === 404) {
        clearInterval(stream.poll);
        handleJobEvent({ type: 'error', message: 'The job disappeared from the server.' });
      }
      return; // transient — keep polling
    }
    if (job.state !== lastState) {
      lastState = job.state;
      if (job.state === 'queued') handleJobEvent({ type: 'queued', queuePosition: job.queuePosition });
      else if (job.state === 'running') handleJobEvent({ type: 'start' });
      else if (job.state === 'decoding') handleJobEvent({ type: 'decoding' });
    }
    if (job.state === 'done') {
      clearInterval(stream.poll);
      handleJobEvent({ type: 'done', ...job });
    } else if (job.state === 'error') {
      clearInterval(stream.poll);
      handleJobEvent({ type: 'error', message: job.error || 'The job failed.' });
    }
  }, 2500);
}

function handleJobEvent(ev) {
  if (!ev || !ev.type) return;

  if (ev.type === 'done' || ev.type === 'error') {
    stream.terminal = true;
    const finishedJobId = stream.jobId;
    const params = state.job && state.job.params;
    teardownStream();
    state.job = null;
    state.submitting = false;
    setRunning(false);

    if (ev.type === 'error') {
      showError(ev.message || 'The pod reported an error but sent no message.');
      loadGallery();
      return;
    }

    showResult({
      jobId: ev.jobId || finishedJobId,
      videoUrl: ev.videoUrl,
      posterUrl: ev.posterUrl,
      durationSec: ev.durationSec,
      width: ev.width,
      height: ev.height,
      seed: ev.seed,
      elapsedSec: ev.elapsedSec,
      steps: params && params.steps,
      prompt: (params && params.prompt) || el.prompt.value,
      mode: (params && params.mode) || state.mode
    });
    if (ev.seed != null) { state.lastSeed = ev.seed; el.seedReuse.disabled = false; saveSettings(); }
    toast(`Done in ${fmtDuration(ev.elapsedSec)} — remember to unmute.`, 'ok', 7000);
    loadGallery();
    return;
  }

  setProgress(ev);
}

/* ───────────────────────── result panel ───────────────────────── */

function showResult(job) {
  state.current = job;
  el.placeholder.hidden = true;
  el.result.hidden = false;
  hideError();

  const v = el.video;
  if (job.posterUrl) v.poster = job.posterUrl;
  v.src = job.videoUrl;
  // Browsers only allow muted autoplay, so we start muted and push the user
  // towards the model's own audio track with an explicit affordance.
  v.muted = true;
  v.load();
  v.play().catch(() => { /* autoplay may still be blocked; controls are there */ });
  el.unmute.hidden = false;

  const frames = job.durationSec != null ? Math.round(job.durationSec * FPS) : null;
  const bits = [
    ['Resolution', job.width && job.height ? `${job.width} × ${job.height}` : '–'],
    ['Duration', job.durationSec != null ? `${Number(job.durationSec).toFixed(2)} s` : '–'],
    ['Frames', frames != null ? `${frames} @ ${FPS} fps` : '–'],
    ['Seed', job.seed != null ? String(job.seed) : '–'],
    ['Steps', job.steps != null ? String(job.steps) : '–'],
    ['Render time', job.elapsedSec != null ? fmtDuration(job.elapsedSec) : '–'],
    ['Mode', MODE_LABELS[job.mode] || job.mode || '–'],
    ['Audio', 'stereo, from the model']
  ];
  el.resultMeta.innerHTML = bits
    .map(([k, val]) => `<span class="meta-item"><span class="meta-k">${escapeHtml(k)}</span><span class="meta-v">${escapeHtml(val)}</span></span>`)
    .join('');

  el.resultPrompt.textContent = job.prompt || '';
  el.resultPrompt.hidden = !job.prompt;

  if (job.videoUrl) {
    el.download.href = job.videoUrl;
    el.download.setAttribute('download', `h3-${job.jobId}.mp4`);
    el.download.removeAttribute('aria-disabled');
  } else {
    el.download.removeAttribute('href');
    el.download.setAttribute('aria-disabled', 'true');
  }

  markActiveCard(job.jobId);
  el.result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

const MODE_LABELS = {
  t2v: 'Text → Video', i2v: 'Image → Video',
  flf2v: 'First + Last frame', r2v: 'Reference → Video'
};

el.unmute.addEventListener('click', () => {
  const v = el.video;
  v.muted = false;
  v.volume = 1;
  v.currentTime = 0;
  v.play().catch(() => { /* ignore */ });
  el.unmute.hidden = true;
});
el.video.addEventListener('volumechange', () => { if (!el.video.muted) el.unmute.hidden = true; });
el.video.addEventListener('error', () => {
  if (el.video.src) toast('The mp4 could not be loaded. It may have been cleaned up on the pod.', 'error');
});

el.copySettings.addEventListener('click', () => {
  const j = state.current;
  if (!j) return;
  if (j.prompt) el.prompt.value = j.prompt;
  if (j.width) el.width.value = j.width;
  if (j.height) el.height.value = j.height;
  if (j.durationSec) el.duration.value = clamp(j.durationSec, 1, 15);
  if (j.steps) el.steps.value = clamp(j.steps, 4, 40);
  if (j.seed != null) el.seed.value = j.seed;
  if (j.mode && el.panes[j.mode]) {
    const tab = el.tabs.find(t => t.dataset.mode === j.mode);
    if (tab && tab.getAttribute('aria-disabled') !== 'true') setMode(j.mode);
  }
  syncAll();
  saveSettings();
  toast('Settings copied into the form. Uploads are not restored — re-add any reference files.', 'ok', 5200);
  el.prompt.scrollIntoView({ behavior: 'smooth', block: 'center' });
});

el.regen.addEventListener('click', () => {
  const j = state.current;
  if (!j) return;
  el.copySettings.click();
  el.seed.value = Math.floor(Math.random() * 4294967295);
  saveSettings();
  setTimeout(() => {
    const reason = blockedReason();
    if (reason) { toast(reason, 'warn'); return; }
    generate();
  }, 120);
});

function showError(message) {
  el.errorbox.hidden = false;
  el.errorboxMsg.textContent = message;
  el.placeholder.hidden = true;
  el.errorbox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
function hideError() { el.errorbox.hidden = true; el.errorboxMsg.textContent = ''; }
el.errorboxDismiss.addEventListener('click', () => {
  hideError();
  if (!state.current) el.placeholder.hidden = false;
});

/* ───────────────────────── gallery ───────────────────────── */

async function loadGallery() {
  let data;
  try {
    data = await api('/api/jobs?limit=30');
  } catch (err) {
    el.galleryEmpty.hidden = false;
    el.galleryEmpty.textContent = `History unavailable: ${err.message}`;
    return;
  }
  state.jobs = (data && data.jobs) || [];
  renderGallery();
}

function renderGallery() {
  el.gallery.innerHTML = '';
  if (!state.jobs.length) {
    el.galleryEmpty.hidden = false;
    el.galleryEmpty.textContent = 'Finished clips land here.';
    return;
  }
  el.galleryEmpty.hidden = true;

  state.jobs.forEach(job => {
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.jobId = job.jobId;
    if (state.current && state.current.jobId === job.jobId) card.classList.add('is-active');

    const thumb = document.createElement('div');
    thumb.className = 'card-thumb';
    if (job.posterUrl) {
      const img = document.createElement('img');
      img.src = job.posterUrl;
      img.alt = '';
      img.loading = 'lazy';
      img.addEventListener('error', () => { img.remove(); addFallback(thumb, job.state); });
      thumb.appendChild(img);
    } else {
      addFallback(thumb, job.state);
    }
    if (job.durationSec != null && job.width) {
      const badge = document.createElement('span');
      badge.className = 'card-badge';
      badge.textContent = `${Number(job.durationSec).toFixed(1)}s · ${job.width}×${job.height}`;
      thumb.appendChild(badge);
    }
    if (job.state && job.state !== 'done') {
      const st = document.createElement('span');
      st.className = 'card-state';
      st.dataset.state = job.state;
      st.textContent = job.state;
      thumb.appendChild(st);
    }
    card.appendChild(thumb);

    const body = document.createElement('div');
    body.className = 'card-body';

    const p = document.createElement('div');
    p.className = 'card-prompt';
    p.textContent = job.prompt || '(no prompt)';
    p.title = job.prompt || '';
    body.appendChild(p);

    const foot = document.createElement('div');
    foot.className = 'card-foot';
    const when = document.createElement('span');
    when.textContent = job.createdAt ? relTime(job.createdAt) : (MODE_LABELS[job.mode] || '');
    when.title = job.createdAt ? new Date(job.createdAt).toLocaleString() : '';
    foot.appendChild(when);

    const actions = document.createElement('div');
    actions.className = 'card-actions';

    const more = document.createElement('button');
    more.type = 'button';
    more.className = 'mini-btn';
    more.textContent = 'prompt';
    more.title = 'Expand the full prompt';
    more.addEventListener('click', ev => {
      ev.stopPropagation();
      p.classList.toggle('is-open');
      more.textContent = p.classList.contains('is-open') ? 'less' : 'prompt';
    });
    actions.appendChild(more);

    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'mini-btn danger';
    del.textContent = 'delete';
    del.setAttribute('aria-label', `Delete job ${job.jobId}`);
    del.addEventListener('click', async ev => {
      ev.stopPropagation();
      del.disabled = true;
      try {
        await api(`/api/jobs/${encodeURIComponent(job.jobId)}`, { method: 'DELETE' });
        state.jobs = state.jobs.filter(j => j.jobId !== job.jobId);
        if (state.current && state.current.jobId === job.jobId) {
          state.current = null;
          el.result.hidden = true;
          el.video.removeAttribute('src');
          el.video.load();
          el.placeholder.hidden = false;
        }
        renderGallery();
      } catch (err) {
        del.disabled = false;
        toast(`Could not delete: ${err.message}`, 'error');
      }
    });
    actions.appendChild(del);
    foot.appendChild(actions);
    body.appendChild(foot);
    card.appendChild(body);

    // whole card opens the clip
    const open = () => {
      if (job.state === 'error') { showError(job.error || 'This job failed.'); return; }
      if (!job.videoUrl) { toast(`That clip is still ${job.state || 'pending'}.`, 'warn'); return; }
      showResult({ ...job, steps: job.steps });
    };
    card.addEventListener('click', open);
    card.tabIndex = 0;
    card.setAttribute('role', 'button');
    card.setAttribute('aria-label', `Open clip: ${truncate(job.prompt || job.jobId, 90)}`);
    card.addEventListener('keydown', ev => {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); open(); }
    });

    el.gallery.appendChild(card);
  });
}

function addFallback(thumb, stateName) {
  const f = document.createElement('span');
  f.className = 'card-fallback';
  f.textContent = stateName === 'error' ? 'failed'
    : (stateName && stateName !== 'done') ? stateName : 'no preview';
  thumb.appendChild(f);
}

function markActiveCard(jobId) {
  $$('.card', el.gallery).forEach(c => c.classList.toggle('is-active', c.dataset.jobId === jobId));
}

function relTime(iso) {
  const t = new Date(iso).getTime();
  if (!isFinite(t)) return '';
  const diff = (Date.now() - t) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

el.refreshGallery.addEventListener('click', loadGallery);

/* ───────────────────────── keyboard ───────────────────────── */

document.addEventListener('keydown', ev => {
  // Cmd/Ctrl + Enter submits from anywhere in the form
  if ((ev.metaKey || ev.ctrlKey) && ev.key === 'Enter') {
    ev.preventDefault();
    const reason = blockedReason();
    if (reason) toast(reason, 'warn');
    else generate();
    return;
  }
  if (ev.key === 'Escape') {
    // close whatever is open, outermost last
    if (!el.structurePanel.hidden) { el.toggleStructure.click(); el.toggleStructure.focus(); return; }
    if (el.advRes.open) { el.advRes.open = false; return; }
    if (!el.errorbox.hidden) { el.errorboxDismiss.click(); return; }
    $$('.card-prompt.is-open').forEach(p => p.classList.remove('is-open'));
  }
});

// Don't let a half-finished job vanish silently on reload
window.addEventListener('beforeunload', ev => {
  if (state.job) { ev.preventDefault(); ev.returnValue = ''; }
});

/* ───────────────────────── go ───────────────────────── */

boot();
