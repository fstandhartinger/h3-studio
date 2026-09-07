/**
 * GPU pod control: rent, watch, extend, stop.
 *
 * The timer and the cost counter tick locally between server updates. That is deliberate
 * — a countdown that only moves when a poll lands looks broken, and the numbers are
 * derived from a server-supplied deadline and rate, so local ticking cannot drift from
 * the truth by more than the poll interval.
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
  bar: $('#podbar'),
  idle: $('#pod-idle'),
  live: $('#pod-live'),
  hours: $('#pod-hours'),
  start: $('#pod-start'),
  stop: $('#pod-stop'),
  extend: $('#pod-extend'),
  dot: $('#pod-dot'),
  phase: $('#pod-phase'),
  timer: $('#pod-timer'),
  cost: $('#pod-cost'),
  msg: $('#pod-msg'),
};

if (el.bar) init();

function init() {
  let pod = null;
  let usdPerHour = 2.09;
  let syncedAt = 0;

  const fmtLeft = (sec) => {
    if (sec == null) return '–';
    const s = Math.max(0, Math.round(sec));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    return h ? `${h}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
             : `${m}:${String(ss).padStart(2, '0')}`;
  };

  function say(text, kind = '') {
    if (!el.msg) return;
    el.msg.textContent = text || '';
    el.msg.dataset.kind = kind;
    el.msg.hidden = !text;
  }

  function paint() {
    const running = pod && pod.state !== 'stopped';
    el.idle.hidden = !!running;
    el.live.hidden = !running;
    if (!running) return;

    // Interpolate between server snapshots so the numbers move every second.
    const drift = (Date.now() - syncedAt) / 1000;
    const elapsed = (pod.elapsedSec ?? 0) + drift;

    el.dot.dataset.state = pod.state;
    el.phase.textContent = pod.state === 'running' ? 'GPU ready' : (pod.phase || pod.state);

    if (pod.secondsLeft == null) {
      // A pod this app did not rent states no budget, and the app does not invent one.
      // Say so plainly instead of showing a countdown that means nothing.
      el.timer.textContent = 'no timer';
      el.timer.parentElement.dataset.level = 'warn';
      el.timer.parentElement.title =
        'This pod was not started here, so it states no deadline. This app will not stop '
        + 'it — the reaper on Sandy still applies its cap. Press +1 h to give it a deadline.';
    } else {
      const left = Math.max(0, pod.secondsLeft - drift);
      el.timer.textContent = fmtLeft(left);
      el.timer.parentElement.dataset.level =
        left < 300 ? 'danger' : left < 900 ? 'warn' : 'ok';
      el.timer.parentElement.title = 'Time until the pod is terminated';
    }

    el.cost.textContent = `$${((elapsed / 3600) * usdPerHour).toFixed(2)}`;
    el.extend.textContent = pod.secondsLeft == null ? 'Set 1 h' : '+1 h';
    el.stop.disabled = false;
  }

  function apply(st) {
    if (!st) return;
    if (st.usdPerHour) usdPerHour = st.usdPerHour;
    pod = st.pod || null;
    syncedAt = Date.now();
    paint();
  }

  setInterval(paint, 1000);

  async function poll() {
    try { apply(await api('/api/pod/status')); } catch { /* transient */ }
  }
  poll();
  setInterval(poll, 15000);

  // Live provisioning progress. The poll above is the fallback if the stream drops.
  function listen() {
    const es = new EventSource('/api/pod/events');
    es.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === 'snapshot') return apply(m);
      if (m.pod) { pod = m.pod; syncedAt = Date.now(); paint(); }
      if (m.type === 'phase') say(`Setting up the pod: ${m.phase}`);
      if (m.type === 'ready') { say('GPU is ready.', 'ok'); poll(); }
      if (m.type === 'error') say(m.message, 'error');
      if (m.type === 'stopped' || m.type === 'gone') { pod = null; paint(); poll(); }
      if (m.type === 'log') console.debug('[pod]', m.msg);
    };
    es.onerror = () => { es.close(); setTimeout(listen, 8000); };
  }
  listen();

  el.start.addEventListener('click', async () => {
    const hours = Number(el.hours.value);
    if (!Number.isFinite(hours) || hours <= 0 || hours > 10) {
      say('Max hours must be between 0.25 and 10.', 'error');
      el.hours.focus();
      return;
    }
    el.start.disabled = true;
    say(`Renting an RTX PRO 6000 for ${hours} h. Installing the models takes about 12 minutes.`);
    try {
      apply(await api('/api/pod/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hours }),
      }));
    } catch (e) {
      say(e.message, 'error');
    } finally {
      el.start.disabled = false;
    }
  });

  el.extend.addEventListener('click', async () => {
    try {
      apply(await api('/api/pod/extend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hours: 1 }),
      }));
      say('Deadline extended by an hour.', 'ok');
    } catch (e) {
      say(e.message, 'error');
    }
  });

  el.stop.addEventListener('click', async () => {
    if (!confirm('Terminate the GPU pod now? Anything still rendering is lost.')) return;
    el.stop.disabled = true;
    say('Terminating…');
    try {
      await api('/api/pod/stop', { method: 'POST' });
      say('Pod terminated. Billing has stopped.', 'ok');
      pod = null;
      paint();
    } catch (e) {
      say(e.message, 'error');
      el.stop.disabled = false;
    }
  });
}
