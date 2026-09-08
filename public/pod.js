/**
 * GPU pod control: rent, watch, extend, stop.
 *
 * The timer and the cost counter tick locally between server updates. That is deliberate
 * — a countdown that only moves when a poll lands looks broken, and the numbers are
 * derived from a server-supplied deadline and rate, so local ticking cannot drift from
 * the truth by more than the poll interval.
 */
import { whenAuthed } from './authed.js';
import { describePhase, fmtMinutes, TYPICAL_MINUTES } from './phases.js';

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
  progress: $('#pod-progress'),
  progressFill: $('#pod-progress-fill'),
  timer: $('#pod-timer'),
  timerBox: $('#pod-timer-box'),
  cost: $('#pod-cost'),
  msg: $('#pod-msg'),
  hint: $('#pod-hint'),
  logBox: $('#pod-log'),
  logLines: $('#pod-log-lines'),
};

// Measured across five installs: 12 min on a good day, 16 on a slow-PyPI day.
const SETUP_MINUTES = 15;

if (el.bar) whenAuthed().then(init);

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

  // The hint under the hours field: what the budget buys, and what it costs.
  function paintHint() {
    const h = Number(el.hours.value);
    if (!Number.isFinite(h) || h <= 0) { el.hint.textContent = 'Enter a budget in hours (up to 10).'; return; }
    const usable = Math.max(0, h * 60 - SETUP_MINUTES);
    el.hint.textContent = usable < 20
      ? `Setup alone takes ~${SETUP_MINUTES} min — a ${h} h budget leaves only ~${Math.round(usable)} min of generation. ≈ $${(h * usdPerHour).toFixed(2)}`
      : `Setup takes ~${SETUP_MINUTES} min of that. ${h} h ≈ ${Math.round(usable)} min of generation · ≈ $${(h * usdPerHour).toFixed(2)}`;
  }
  el.hours.addEventListener('input', paintHint);

  function paintLog(lines) {
    if (!Array.isArray(lines) || !lines.length) { el.logBox.hidden = true; return; }
    el.logBox.hidden = false;
    el.logLines.replaceChildren(...lines.slice(-10).map((l) => {
      const li = document.createElement('li');
      const t = document.createElement('time');
      t.dateTime = new Date(l.t).toISOString();
      t.textContent = new Date(l.t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      li.append(t, document.createTextNode(l.msg));
      return li;
    }));
  }

  function paint() {
    const running = pod && pod.state !== 'stopped';
    el.idle.hidden = !!running;
    el.live.hidden = !running;
    el.hint.hidden = !!running;
    if (!running) { paintHint(); return; }

    // Interpolate between server snapshots so the numbers move every second.
    const drift = (Date.now() - syncedAt) / 1000;
    const elapsed = (pod.elapsedSec ?? 0) + drift;

    el.dot.dataset.state = pod.state;
    if (pod.state === 'provisioning') {
      const ph = describePhase(pod.phase);
      el.phase.textContent = `Setting up · ${ph.step}/${ph.total} ${ph.label} · ${fmtMinutes(elapsed)}`;
      el.phase.title = `Usually ${TYPICAL_MINUTES} min in total`;
      el.progress.hidden = false;
      el.progressFill.style.width = `${ph.pct}%`;
    } else {
      el.phase.textContent = pod.state === 'running' ? 'GPU ready'
        : pod.state === 'error' ? 'Setup failed — see Pod activity' : (pod.phase || pod.state);
      el.phase.title = '';
      el.progress.hidden = true;
    }

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
      el.timer.parentElement.title =
        `Terminates at ${new Date(pod.deadline).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
        + ' unless extended';
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
    if (st.log) paintLog(st.log);
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
      if (m.type === 'ready') {
        // A 1 h budget minus a 15 min install is 45 min of use; say so while it can
        // still be fixed with one click, not when the timer hits zero mid-render.
        const left = m.pod?.secondsLeft ?? pod?.secondsLeft;
        if (left != null && left < 45 * 60) {
          say(`GPU is ready — but only ${fmtMinutes(left)} remain on the budget. Press +1 h if you need longer.`, 'warn');
        } else {
          say('GPU is ready.', 'ok');
        }
        poll();
      }
      if (m.type === 'log' && m.msg) {
        // keep the visible log live between polls
        const li = document.createElement('li');
        const t = document.createElement('time');
        t.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        li.append(t, document.createTextNode(m.msg));
        el.logBox.hidden = false;
        el.logLines.append(li);
        while (el.logLines.children.length > 10) el.logLines.firstChild.remove();
      }
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
    say(`Renting an RTX PRO 6000 for ${hours} h. Installing the models takes about ${TYPICAL_MINUTES} minutes — `
      + 'the bar above shows each step.');
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
