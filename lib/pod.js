/**
 * RunPod lifecycle: rent a GPU, install the stack on it, watch the clock, stop it.
 *
 * The deadline is the important part. A GPU at $2.09/h is $50/day, so every pod this
 * module creates carries its expiry in TWO independent places:
 *
 *   1. `H3_DEADLINE` in the pod's own environment, readable back from the RunPod API.
 *      Immutable after creation, and the value the Sandy reaper falls back to.
 *   2. This module's own timer, which is what an "extend" actually moves.
 *
 * They are separate because RunPod restarts a container when its environment is patched,
 * which would kill whatever is rendering. So an extension cannot be written back to the
 * pod; it lives here and is published at /api/pod/deadlines for the reaper to read.
 * If this app is down, the reaper falls back to (1), which is never longer than the
 * originally requested budget. Every failure mode ends with the pod dying earlier than
 * intended rather than later.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fsp from 'node:fs/promises';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const execFileP = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const API = 'https://rest.runpod.io/v1/pods';
const GRAPHQL = 'https://api.runpod.io/graphql';
const GPU = 'NVIDIA RTX PRO 6000 Blackwell Server Edition';
const IMAGE = 'runpod/pytorch:1.1.0-rc.154-cu1290-torch291-ubuntu2404';

// RunPod sits behind Cloudflare, which answers "403 error code: 1010" to the default
// Node/undici and Python User-Agents. Any ordinary UA gets through.
const UA = 'curl/8.5.0';

const KEY = () => process.env.RUNPOD_API_KEY || '';
export const podControlConfigured = () =>
  !!(KEY() && process.env.POD_SSH_PRIVATE_KEY && process.env.POD_SSH_PUBLIC_KEY);

export const MAX_HOURS = 10;
export const DEFAULT_HOURS = 1;
export const USD_PER_HOUR = Number(process.env.GPU_USD_PER_HOUR || 2.09);

/** In-memory view of the pod this app manages. */
let current = null;   // { id, deadline, hours, state, phase, createdAt, error, comfyUrl }
const log = [];
const listeners = new Set();

function emit(ev) {
  const line = `data: ${JSON.stringify(ev)}\n\n`;
  for (const res of listeners) { try { res.write(line); } catch { /* gone */ } }
}

function note(msg) {
  log.push({ t: Date.now(), msg });
  while (log.length > 200) log.shift();
  emit({ type: 'log', msg });
}

export function subscribe(res) {
  listeners.add(res);
  return () => listeners.delete(res);
}

async function rp(pathname = '', { method = 'GET', body } = {}) {
  const r = await fetch(API + pathname, {
    method,
    headers: {
      Authorization: `Bearer ${KEY()}`,
      'Content-Type': 'application/json',
      'User-Agent': UA,
      Accept: 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  let json = null;
  if (text) { try { json = JSON.parse(text); } catch { /* non-JSON */ } }
  if (!r.ok) {
    throw new Error(`RunPod ${method} ${pathname || '/'} -> ${r.status} ${text.slice(0, 200)}`);
  }
  return json;
}

/** Go's time.String() — "2026-09-07 20:50:22.284 +0000 UTC" — which Date cannot parse. */
export function parseRunpodTime(v) {
  if (!v) return null;
  if (typeof v === 'number') return v > 1e11 ? v : v * 1000;
  let s = String(v).trim()
    .replace(/\s+[A-Z]{2,5}$/, '')          // trailing zone NAME
    .replace(/\s+([+-]\d{2}:?\d{2})$/, '$1') // space before the offset
    .replace(/([+-]\d{2})(\d{2})$/, '$1:$2') // +0000 -> +00:00
    .replace('Z', '+00:00');
  let t = Date.parse(s);
  if (Number.isNaN(t)) t = Date.parse(s.replace(' ', 'T'));
  return Number.isNaN(t) ? null : t;
}

export async function listPods() {
  const pods = await rp();
  return Array.isArray(pods) ? pods : (pods?.data || []);
}

/**
 * Reconcile in-memory state with RunPod. Called on startup and by /api/pod/status, so a
 * container restart re-adopts a pod that is still running instead of losing track of it.
 */
export async function refresh() {
  if (!KEY()) return null;
  let pods;
  try { pods = await listPods(); } catch (e) { note(`RunPod unreachable: ${e.message}`); return current; }

  const mine = pods.find((p) => p.env?.H3_MANAGED === '1') || pods[0] || null;
  if (!mine) {
    if (current && current.state !== 'stopped') {
      note(`pod ${current.id} is gone`);
      current = null;
      emit({ type: 'gone' });
    }
    return null;
  }

  const createdAt = parseRunpodTime(mine.createdAt) || Date.now();
  const envDeadline = Number(mine.env?.H3_DEADLINE) * 1000 || null;

  if (!current || current.id !== mine.id) {
    // Adopting a pod this process did not create: a container restart, or one of the CLI
    // scripts. Only a pod carrying H3_DEADLINE states its own budget; anything else has a
    // deadline this app does not know.
    //
    // This app must NOT enforce a deadline it had to invent. An early version assumed
    // DEFAULT_HOURS from createdAt, adopted an hour-old pod, computed a deadline already
    // in the past, and terminated a pod that was mid-render one second after starting up.
    // Guessing wrong here destroys work, so an unknown budget is now carried as unknown
    // and left to the Sandy reaper, which is the component that is actually allowed to
    // apply a blanket cap.
    const adopted = !envDeadline;
    current = {
      id: mine.id,
      createdAt,
      deadline: envDeadline || null,
      hours: envDeadline ? (envDeadline - createdAt) / 3600_000 : null,
      unmanaged: adopted,
      state: 'running',
      phase: adopted ? 'running (deadline not set here)' : 'adopted',
      comfyUrl: `https://${mine.id}-8188.proxy.runpod.net`,
      costPerHr: Number(mine.costPerHr) || USD_PER_HOUR,
    };
    note(adopted
      ? `adopted pod ${mine.id}, which states no deadline — this app will not stop it; `
        + 'the Sandy reaper still applies its cap'
      : `adopted pod ${mine.id}, deadline ${new Date(envDeadline).toISOString()}`);
  } else {
    current.costPerHr = Number(mine.costPerHr) || current.costPerHr;
    current.createdAt = createdAt;
  }
  return current;
}

export function status() {
  if (!current) {
    return { pod: null, configured: podControlConfigured(), maxHours: MAX_HOURS,
             defaultHours: DEFAULT_HOURS, usdPerHour: USD_PER_HOUR, log: log.slice(-40) };
  }
  const now = Date.now();
  const elapsedSec = Math.max(0, Math.round((now - current.createdAt) / 1000));
  const secondsLeft = current.deadline == null
    ? null : Math.max(0, Math.round((current.deadline - now) / 1000));
  return {
    configured: podControlConfigured(),
    maxHours: MAX_HOURS,
    defaultHours: DEFAULT_HOURS,
    usdPerHour: current.costPerHr || USD_PER_HOUR,
    pod: {
      id: current.id,
      state: current.state,
      phase: current.phase,
      error: current.error || undefined,
      comfyUrl: current.comfyUrl,
      createdAt: current.createdAt,
      deadline: current.deadline,
      secondsLeft,
      unmanaged: !!current.unmanaged,
      elapsedSec,
      // Billing starts when the pod is rented, not when it becomes useful, so the
      // counter deliberately includes the provisioning minutes.
      costSoFar: +((elapsedSec / 3600) * (current.costPerHr || USD_PER_HOUR)).toFixed(3),
      hours: current.hours,
    },
    log: log.slice(-40),
  };
}

/** The deadlines the Sandy reaper asks about. */
export function deadlines() {
  if (!current || current.state === 'stopped' || current.deadline == null) return {};
  return { [current.id]: { deadline: Math.round(current.deadline / 1000), source: 'h3-studio' } };
}

// ------------------------------------------------------------------ ssh

function keyPath() {
  const p = path.join(os.tmpdir(), 'h3-pod-key');
  if (!fs.existsSync(p)) {
    let k = process.env.POD_SSH_PRIVATE_KEY || '';
    // Env vars cannot hold real newlines through every layer, so \n is accepted too.
    k = k.replace(/\\n/g, '\n');
    if (!k.endsWith('\n')) k += '\n';
    fs.writeFileSync(p, k, { mode: 0o600 });
  }
  return p;
}

const SSH_OPTS = [
  '-o', 'StrictHostKeyChecking=no',
  '-o', 'UserKnownHostsFile=/dev/null',
  '-o', 'ConnectTimeout=15',
  '-o', 'LogLevel=ERROR',
];

async function sshEndpoint(podId) {
  const r = await fetch(GRAPHQL, {
    method: 'POST',
    headers: { Authorization: `Bearer ${KEY()}`, 'Content-Type': 'application/json', 'User-Agent': UA },
    body: JSON.stringify({
      query: `query { pod(input:{podId:"${podId}"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }`,
    }),
  });
  const j = await r.json().catch(() => null);
  const ports = j?.data?.pod?.runtime?.ports || [];
  const ssh = ports.find((p) => p.privatePort === 22 && p.isIpPublic);
  return ssh ? { ip: ssh.ip, port: ssh.publicPort } : null;
}

async function ssh(ep, cmd, timeout = 120000) {
  return execFileP('ssh', [...SSH_OPTS, '-i', keyPath(), '-p', String(ep.port),
    `root@${ep.ip}`, cmd], { timeout, maxBuffer: 8 * 1024 * 1024 });
}

async function scp(ep, local, remote) {
  return execFileP('scp', [...SSH_OPTS, '-i', keyPath(), '-P', String(ep.port),
    local, `root@${ep.ip}:${remote}`], { timeout: 120000 });
}

// ------------------------------------------------------------- lifecycle

export async function start({ hours = DEFAULT_HOURS, chroma = true } = {}) {
  if (!podControlConfigured()) throw new Error('pod control is not configured on this deployment');
  await refresh();
  if (current && current.state !== 'stopped') throw new Error(`pod ${current.id} is already running`);

  const h = Math.min(MAX_HOURS, Math.max(0.25, Number(hours) || DEFAULT_HOURS));
  const createdAt = Date.now();
  const deadline = createdAt + h * 3600_000;

  note(`renting an RTX PRO 6000 for ${h} h`);
  const body = {
    name: `h3-studio-${new Date().toISOString().slice(11, 16).replace(':', '')}`,
    imageName: IMAGE,
    cloudType: 'SECURE',
    computeType: 'GPU',
    gpuTypeIds: [GPU],
    gpuCount: 1,
    containerDiskInGb: 100,
    volumeInGb: 250,
    volumeMountPath: '/workspace',
    minRAMPerGPU: 150,
    minVCPUPerGPU: 16,
    minDownloadMbps: 1000,
    ports: ['8188/http', '22/tcp'],
    supportPublicIp: true,
    env: {
      PUBLIC_KEY: process.env.POD_SSH_PUBLIC_KEY || '',
      H3_MANAGED: '1',
      // The reaper's fallback if this app is unreachable. Never longer than the budget.
      H3_DEADLINE: String(Math.round(deadline / 1000)),
    },
  };

  const created = await rp('', { method: 'POST', body });
  const id = created?.id;
  if (!id) throw new Error(`RunPod did not return a pod id: ${JSON.stringify(created).slice(0, 200)}`);

  current = {
    id, createdAt, deadline, hours: h,
    state: 'provisioning', phase: 'booting',
    comfyUrl: `https://${id}-8188.proxy.runpod.net`,
    costPerHr: USD_PER_HOUR,
  };
  note(`pod ${id} created, deadline ${new Date(deadline).toISOString()}`);
  emit({ type: 'started', pod: status().pod });

  provision(id, { chroma }).catch((e) => {
    if (current?.id === id) {
      current.state = 'error';
      current.error = e.message;
      current.phase = 'failed';
    }
    note(`provisioning failed: ${e.message}`);
    emit({ type: 'error', message: e.message });
  });

  armTimer();
  return status();
}

async function provision(id, { chroma }) {
  const setPhase = (p) => {
    if (current?.id !== id) return;
    current.phase = p;
    emit({ type: 'phase', phase: p, pod: status().pod });
  };

  setPhase('waiting for ssh');
  let ep = null;
  for (let i = 0; i < 60 && current?.id === id; i++) {
    ep = await sshEndpoint(id).catch(() => null);
    if (ep) break;
    await new Promise((r) => setTimeout(r, 10000));
  }
  if (!ep) throw new Error('pod never exposed an ssh port');
  note(`ssh up at ${ep.ip}:${ep.port}`);

  for (let i = 0; i < 30; i++) {
    try { await ssh(ep, 'true', 20000); break; } catch {
      if (i === 29) throw new Error('ssh never accepted a connection');
      await new Promise((r) => setTimeout(r, 10000));
    }
  }

  setPhase('uploading installer');
  await scp(ep, path.join(__dirname, '..', 'provision', 'pdl.py'), '/root/pdl.py');
  await scp(ep, path.join(__dirname, '..', 'provision', 'setup.sh'), '/root/setup.sh');

  setPhase('installing');
  // setsid + nohup so the installer outlives this ssh connection; it runs ~12 minutes
  // and a dropped connection must not take it down.
  await ssh(ep, `chmod +x /root/setup.sh && WANT_CHROMA=${chroma ? 1 : 0} `
              + 'setsid nohup /root/setup.sh >/dev/null 2>&1 </dev/null & echo started', 30000);

  // Poll the phase file rather than streaming stdout: it survives a dropped connection.
  let last = '';
  for (let i = 0; i < 240 && current?.id === id; i++) {
    await new Promise((r) => setTimeout(r, 10000));
    let phase = '';
    try {
      const { stdout } = await ssh(ep, 'cat /workspace/phase 2>/dev/null || true', 25000);
      phase = stdout.trim();
    } catch { continue; }
    if (phase && phase !== last) {
      last = phase;
      setPhase(phase);
      note(`pod phase: ${phase}`);
    }
    if (phase === 'ready') {
      if (current?.id === id) {
        current.state = 'running';
        current.phase = 'ready';
      }
      note('pod is ready');
      emit({ type: 'ready', pod: status().pod });
      return;
    }
    if (phase === 'failed') throw new Error('the installer reported failure; see /workspace/setup.log on the pod');
  }
  throw new Error('provisioning timed out');
}

export async function stop(reason = 'stopped from the web app') {
  if (!current) throw new Error('no pod is running');
  const id = current.id;
  note(`terminating ${id}: ${reason}`);
  let gone = false;
  for (let i = 0; i < 3 && !gone; i++) {
    try {
      await rp(`/${id}`, { method: 'DELETE' });
      gone = true;
    } catch (e) {
      if (/-> 404/.test(e.message)) { gone = true; break; }
      await new Promise((r) => setTimeout(r, 5000));
    }
  }
  // Trust the verification, not the delete response.
  try {
    await rp(`/${id}`);
    gone = false;
  } catch (e) { if (/-> 404/.test(e.message)) gone = true; }

  if (current?.id === id) {
    current.state = gone ? 'stopped' : 'error';
    current.phase = gone ? 'stopped' : 'still present — check RunPod';
    if (!gone) current.error = 'delete did not take effect';
  }
  emit({ type: 'stopped', ok: gone });
  if (gone) current = null;
  return { ok: gone };
}

export function extend(hours) {
  if (!current || current.state === 'stopped') throw new Error('no pod is running');
  const add = Number(hours);
  if (!Number.isFinite(add) || add <= 0) throw new Error('hours must be a positive number');
  // Extending a pod with no known deadline is how you give it one: from now, not from a
  // creation time whose budget nobody recorded.
  if (current.deadline == null) {
    current.deadline = Date.now();
    current.unmanaged = false;
  }
  const total = (current.deadline + add * 3600_000 - current.createdAt) / 3600_000;
  if (total > MAX_HOURS) {
    throw new Error(`that would make the pod run ${total.toFixed(1)} h; the cap is ${MAX_HOURS} h`);
  }
  current.deadline += add * 3600_000;
  current.hours = total;
  note(`deadline extended to ${new Date(current.deadline).toISOString()}`);
  armTimer();
  emit({ type: 'extended', pod: status().pod });
  return status();
}

/**
 * The app's own enforcement. The Sandy cron reaper is the independent net that catches
 * this app being down; this timer is what makes an extension take effect without a
 * container restart.
 */
let timer = null;
function armTimer() {
  if (timer) clearTimeout(timer);
  timer = null;
  if (!current || current.state === 'stopped') return;
  if (current.deadline == null) return;   // unknown budget: not this app's to enforce

  const ms = current.deadline - Date.now();
  if (ms <= 0) {
    // Already past. This is only reachable for a pod THIS app rented, where the deadline
    // is a fact rather than a guess -- but stopping instantly on a startup race would
    // still be indistinguishable from the bug above, so it is logged and given a minute.
    note('deadline is already past; stopping in 60 s');
    timer = setTimeout(() => stop('deadline reached')
      .catch((e) => note(`auto-stop failed: ${e.message}`)), 60000);
    timer.unref?.();
    return;
  }
  timer = setTimeout(() => {
    note('deadline reached');
    stop('deadline reached').catch((e) => note(`auto-stop failed: ${e.message}`));
  }, Math.min(ms, 2 ** 31 - 1));
  timer.unref?.();
}

export async function init() {
  if (!KEY()) return;
  await refresh().catch(() => {});
  armTimer();
  setInterval(() => { refresh().catch(() => {}); }, 60000).unref?.();
}
