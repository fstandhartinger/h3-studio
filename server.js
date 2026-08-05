/**
 * H3 Studio — a thin, stateful front door to a MiniMax H3 ComfyUI instance
 * running on a remote RunPod GPU.
 *
 * Responsibilities that genuinely belong here rather than in the browser:
 *   - hold the single ComfyUI websocket and fan its progress events out as SSE
 *   - build the API-format ComfyUI graphs (the shipped templates are UI-format
 *     subgraphs and cannot be POSTed to /prompt)
 *   - cache finished videos locally so the gallery survives the pod being killed
 *   - keep the ComfyUI endpoint off the public internet behind one shared password
 */
import express from 'express';
import cookieParser from 'cookie-parser';
import multer from 'multer';
import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import WebSocket from 'ws';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------- config

const PORT = process.env.PORT || 3000;
const COMFY_URL = (process.env.COMFY_URL || '').replace(/\/+$/, '');
const APP_PASSWORD = process.env.APP_PASSWORD || '';
const SESSION_SECRET = process.env.SESSION_SECRET || crypto.randomBytes(32).toString('hex');
const POD_DEADLINE = process.env.POD_DEADLINE || ''; // ISO8601, optional
const CACHE_DIR = process.env.CACHE_DIR || path.join('/tmp', 'h3studio-cache');
const CACHE_MAX_BYTES = Number(process.env.CACHE_MAX_BYTES || 250 * 1024 * 1024);
const MAX_HISTORY = 60;

const FPS = 24;
const GPU_USD_PER_HOUR = Number(process.env.GPU_USD_PER_HOUR || 2.09);

fs.mkdirSync(CACHE_DIR, { recursive: true });

/** ComfyUI checkpoint filenames (from Comfy-Org/MiniMax-H3). */
const CKPT = {
  fl2va: 'minimax_h3_fl2va_pruned_int8_convrot.safetensors',
  ref2va: 'minimax_h3_ref2va_pruned_int8_convrot.safetensors',
  clip: 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
  videoVae: 'minimax_h3_video_vae_fp16.safetensors',
  audioVae: 'minimax_h3_audio_vae_fp32.safetensors',
};

// ------------------------------------------------------- frame-grid maths

/**
 * The H3 video VAE (f16t4d24, 1x2x2 patchification) only accepts frame counts
 * where `length % 17 == 5`. ComfyUI declares this as min=5, step=17.
 * Snap a requested duration down to the nearest legal count.
 */
export function snapFrames(seconds) {
  const wanted = Math.max(5, Math.round(Number(seconds) * FPS));
  let best = 5;
  for (let l = 5; l <= 3600; l += 17) {
    if (l <= wanted) best = l;
    else break;
  }
  return best;
}

/**
 * Rough wall-clock estimate. Calibrated on an RTX PRO 6000 Blackwell at 20 steps:
 * 124 frames @1344x768 -> 370 s; 243 frames @1344x768 -> 917 s.
 * Attention is quadratic in sequence length, so cost grows faster than linearly
 * in (pixels x frames); ~1.35 fits both measured points well enough to be useful.
 */
export function estimateSeconds({ width, height, frames, steps }) {
  const refTokens = 1344 * 768 * 124;
  const tokens = width * height * frames;
  const perStepAtRef = 370 / 20;
  const scale = Math.pow(tokens / refTokens, 1.35);
  return Math.round(perStepAtRef * scale * steps + 25); // +25s load/decode floor
}

// -------------------------------------------------------- ComfyUI client

const comfyHttp = (p) => `${COMFY_URL}${p}`;

async function comfyFetch(p, opts = {}, timeoutMs = 30000) {
  if (!COMFY_URL) throw new Error('COMFY_URL is not configured');
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(comfyHttp(p), { ...opts, signal: ctrl.signal });
    return r;
  } finally {
    clearTimeout(t);
  }
}

async function comfyJson(p, opts, timeoutMs) {
  const r = await comfyFetch(p, opts, timeoutMs);
  if (!r.ok) throw new Error(`ComfyUI ${p} -> ${r.status}`);
  return r.json();
}

// ------------------------------------------------------- graph building

let nodeSeq = 0;
const nid = () => String(++nodeSeq);

/**
 * Build an API-format ComfyUI graph. Mirrors the shipped
 * video_minimax_h3_{t2v,i2v,r2v} templates with the subgraph flattened.
 */
function buildGraph(job) {
  const { mode, prompt, width, height, length, steps, seed } = job;
  const isRef = mode === 'r2v';
  const g = {};
  const add = (class_type, inputs) => {
    const id = nid();
    g[id] = { class_type, inputs };
    return id;
  };

  const unet = add('UNETLoader', {
    unet_name: isRef ? CKPT.ref2va : CKPT.fl2va,
    weight_dtype: 'default',
  });
  const clip = add('CLIPLoader', { clip_name: CKPT.clip, type: 'minimax', device: 'default' });
  const vVae = add('VAELoader', { vae_name: CKPT.videoVae });
  const aVae = add('VAELoader', { vae_name: CKPT.audioVae });

  let cond;
  if (isRef) {
    const inputs = {
      clip: [clip, 0],
      vae: [vVae, 0],
      audio_vae: [aVae, 0],
      prompt,
      width,
      height,
      length,
      ref_image_size: job.refImageSize || 'match',
    };
    // COMFY_AUTOGROW_V3 inputs are addressed as "<group>.<prefix><n>".
    (job.refImages || []).slice(0, 9).forEach((name, i) => {
      const n = add('LoadImage', { image: name, upload: 'image' });
      inputs[`ref_images.ref_image_${i}`] = [n, 0];
    });
    (job.refVideos || []).slice(0, 3).forEach((name, i) => {
      const lv = add('LoadVideo', { file: name });
      const gc = add('GetVideoComponents', { video: [lv, 0] });
      inputs[`ref_videos.ref_video_${i}`] = [gc, 0];
      // carry the clip's own soundtrack through as the matching reference audio
      inputs[`ref_video_audios.ref_video_audio_${i}`] = [gc, 1];
    });
    (job.refAudios || []).slice(0, 3).forEach((name, i) => {
      const n = add('LoadAudio', { audio: name });
      inputs[`ref_audios.ref_audio_${i}`] = [n, 0];
    });
    cond = add('MiniMaxH3ReferenceToVideo', inputs);
  } else {
    const inputs = { clip: [clip, 0], vae: [vVae, 0], prompt, width, height, length };
    if (job.firstFrame) {
      const n = add('LoadImage', { image: job.firstFrame, upload: 'image' });
      inputs.first_frame = [n, 0];
    }
    if (job.lastFrame) {
      const n = add('LoadImage', { image: job.lastFrame, upload: 'image' });
      inputs.last_frame = [n, 0];
    }
    cond = add('MiniMaxH3ImageToVideo', inputs);
  }

  const noise = add('RandomNoise', { noise_seed: seed });
  const guider = add('BasicGuider', { model: [unet, 0], conditioning: [cond, 0] });
  // The released checkpoints are CFG-distilled -> BasicGuider, not CFGGuider.
  const sampler = add('KSamplerSelect', { sampler_name: 'res_multistep' });
  const sigmas = add('BasicScheduler', {
    model: [unet, 0], scheduler: 'simple', steps, denoise: 1.0,
  });
  const sampled = add('SamplerCustomAdvanced', {
    noise: [noise, 0], guider: [guider, 0], sampler: [sampler, 0],
    sigmas: [sigmas, 0], latent_image: [cond, 1],
  });

  const frames = add('VAEDecode', { samples: [sampled, 0], vae: [vVae, 0] });
  const audio = add('VAEDecodeAudio', { samples: [sampled, 0], vae: [aVae, 0] });
  const video = add('CreateVideo', { images: [frames, 0], audio: [audio, 0], fps: FPS });
  const save = add('SaveVideo', {
    video: [video, 0], filename_prefix: `h3studio/${job.id}`, format: 'auto', codec: 'auto',
  });

  // Poster straight out of the graph: first decoded frame -> SaveImage.
  // Cheaper and more portable than shipping ffmpeg with the web app.
  const one = add('ImageFromBatch', { image: [frames, 0], batch_index: 0, length: 1 });
  const poster = add('SaveImage', { images: [one, 0], filename_prefix: `h3poster/${job.id}` });

  return { graph: g, saveNode: save, posterNode: poster };
}

// ------------------------------------------------------------ job store

/** @type {Map<string, any>} */
const jobs = new Map();
const order = []; // newest-first job ids
const listeners = new Map(); // jobId -> Set<res>

function pushJob(job) {
  jobs.set(job.id, job);
  order.unshift(job.id);
  while (order.length > MAX_HISTORY) {
    const gone = order.pop();
    const j = jobs.get(gone);
    if (j?.cachedVideo) fsp.unlink(j.cachedVideo).catch(() => {});
    if (j?.cachedPoster) fsp.unlink(j.cachedPoster).catch(() => {});
    jobs.delete(gone);
  }
}

function emit(jobId, payload) {
  const set = listeners.get(jobId);
  if (!set) return;
  const line = `data: ${JSON.stringify(payload)}\n\n`;
  for (const res of set) {
    try { res.write(line); } catch { /* client vanished */ }
  }
  if (payload.type === 'done' || payload.type === 'error') {
    for (const res of set) { try { res.end(); } catch {} }
    listeners.delete(jobId);
  }
}

function publicJob(j) {
  return {
    jobId: j.id, state: j.state, mode: j.mode, prompt: j.prompt,
    width: j.width, height: j.height, frames: j.length,
    durationSec: +(j.length / FPS).toFixed(2), steps: j.steps, seed: j.seed,
    createdAt: j.createdAt, elapsedSec: j.elapsedSec ?? null,
    step: j.step ?? 0, totalSteps: j.steps,
    error: j.error || undefined,
    videoUrl: j.state === 'done' ? `/api/video/${j.id}` : undefined,
    posterUrl: j.state === 'done' && j.posterFile ? `/api/poster/${j.id}` : undefined,
  };
}

// ------------------------------------------- ComfyUI websocket -> SSE

const CLIENT_ID = crypto.randomUUID();
let ws = null;
let wsAlive = false;
const promptToJob = new Map();

function connectWs() {
  if (!COMFY_URL) return;
  const url = COMFY_URL.replace(/^http/, 'ws') + `/ws?clientId=${CLIENT_ID}`;
  try {
    ws = new WebSocket(url, { handshakeTimeout: 15000 });
  } catch {
    return setTimeout(connectWs, 5000);
  }
  ws.on('open', () => { wsAlive = true; });
  ws.on('close', () => { wsAlive = false; ws = null; setTimeout(connectWs, 5000); });
  ws.on('error', () => { /* close handler schedules the retry */ });
  ws.on('message', (raw, isBinary) => {
    if (isBinary) return; // preview images, ignored
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }
    const d = msg.data || {};
    const jobId = d.prompt_id ? promptToJob.get(d.prompt_id) : null;
    if (!jobId) return;
    const job = jobs.get(jobId);
    if (!job || job.state === 'done' || job.state === 'error') return;

    if (msg.type === 'execution_start') {
      job.state = 'running';
      job.startedAt = Date.now();
      emit(jobId, { type: 'start' });
    } else if (msg.type === 'progress') {
      const step = d.value, total = d.max || job.steps;
      job.state = 'running';
      job.step = step;
      if (step > 0 && job.startedAt) {
        const per = (Date.now() - job.startedAt) / 1000 / step;
        job.etaSeconds = Math.max(0, Math.round(per * (total - step)));
      }
      emit(jobId, {
        type: 'progress', step, totalSteps: total,
        pct: total ? (step / total) * 100 : 0,
        etaSeconds: job.etaSeconds ?? null,
      });
      // The sampler is the only thing reporting steps; once it tops out the
      // remaining time is VAE decode + mux, which is silent.
      if (step >= total) {
        job.state = 'decoding';
        emit(jobId, { type: 'decoding' });
      }
    } else if (msg.type === 'executing' && d.node === null) {
      finishJob(jobId).catch((e) => failJob(jobId, e.message));
    } else if (msg.type === 'execution_error') {
      failJob(jobId, d.exception_message || 'execution error');
    } else if (msg.type === 'execution_interrupted') {
      failJob(jobId, 'cancelled');
    }
  });
}

function failJob(jobId, message) {
  const job = jobs.get(jobId);
  if (!job || job.state === 'done' || job.state === 'error') return;
  job.state = 'error';
  job.error = message;
  emit(jobId, { type: 'error', message });
}

/** Pull the finished mp4 + poster out of ComfyUI and cache them locally. */
async function finishJob(jobId) {
  const job = jobs.get(jobId);
  if (!job || job.state === 'done') return;
  const hist = await comfyJson(`/history/${job.promptId}`, {}, 60000);
  const entry = hist[job.promptId];
  if (!entry) throw new Error('job finished but history is empty');

  const status = entry.status || {};
  if (status.status_str === 'error') {
    const m = (status.messages || []).find((x) => x[0] === 'execution_error');
    throw new Error(m ? JSON.stringify(m[1]).slice(0, 400) : 'execution error');
  }

  const outs = entry.outputs || {};
  const pick = (nodeId, keys) => {
    const o = outs[nodeId] || {};
    for (const k of keys) if (o[k]?.length) return o[k][0];
    return null;
  };
  const vid = pick(job.saveNode, ['images', 'videos', 'gifs']);
  if (!vid) throw new Error('no video in ComfyUI output');
  const pos = pick(job.posterNode, ['images']);

  job.cachedVideo = await cacheFile(vid, `${job.id}.mp4`);
  if (pos) job.cachedPoster = await cacheFile(pos, `${job.id}.png`);
  job.posterFile = !!job.cachedPoster;

  job.state = 'done';
  job.elapsedSec = job.startedAt ? Math.round((Date.now() - job.startedAt) / 1000) : null;
  await pruneCache();
  emit(jobId, {
    type: 'done', jobId, videoUrl: `/api/video/${jobId}`,
    posterUrl: job.posterFile ? `/api/poster/${jobId}` : null,
    durationSec: +(job.length / FPS).toFixed(2),
    width: job.width, height: job.height, seed: job.seed,
    elapsedSec: job.elapsedSec,
  });
}

async function cacheFile(ref, localName) {
  const q = new URLSearchParams({
    filename: ref.filename, subfolder: ref.subfolder || '', type: ref.type || 'output',
  });
  const r = await comfyFetch(`/view?${q}`, {}, 180000);
  if (!r.ok) throw new Error(`fetching ${ref.filename} -> ${r.status}`);
  const dest = path.join(CACHE_DIR, localName);
  await fsp.writeFile(dest, Buffer.from(await r.arrayBuffer()));
  return dest;
}

async function pruneCache() {
  try {
    const names = await fsp.readdir(CACHE_DIR);
    const stats = await Promise.all(names.map(async (n) => {
      const p = path.join(CACHE_DIR, n);
      const s = await fsp.stat(p);
      return { p, size: s.size, mtime: s.mtimeMs };
    }));
    let total = stats.reduce((a, b) => a + b.size, 0);
    stats.sort((a, b) => a.mtime - b.mtime);
    for (const f of stats) {
      if (total <= CACHE_MAX_BYTES) break;
      await fsp.unlink(f.p).catch(() => {});
      total -= f.size;
    }
  } catch { /* cache pruning is best-effort */ }
}

/**
 * Websocket events can be missed (reconnects, Render idling the dyno).
 * A slow poll over anything still in flight is the safety net.
 */
setInterval(async () => {
  for (const id of order) {
    const j = jobs.get(id);
    if (!j || (j.state !== 'running' && j.state !== 'queued' && j.state !== 'decoding')) continue;
    if (Date.now() - j.createdAt > 3 * 3600 * 1000) { failJob(id, 'timed out'); continue; }
    try {
      const hist = await comfyJson(`/history/${j.promptId}`, {}, 20000);
      if (hist[j.promptId]?.status?.completed) await finishJob(id);
      else if (hist[j.promptId]?.status?.status_str === 'error') await finishJob(id).catch((e) => failJob(id, e.message));
    } catch { /* pod may be down; the status endpoint reports that separately */ }
  }
}, 20000).unref();

connectWs();

// ---------------------------------------------------------------- app

const app = express();
app.set('trust proxy', 1);
app.use(express.json({ limit: '2mb' }));
app.use(cookieParser(SESSION_SECRET));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 120 * 1024 * 1024 },
});

const authRequired = !!APP_PASSWORD;
const sessionToken = () =>
  crypto.createHmac('sha256', SESSION_SECRET).update('h3studio-v1').digest('hex');

function requireAuth(req, res, next) {
  if (!authRequired) return next();
  if (req.signedCookies?.h3s === sessionToken()) return next();
  return res.status(401).json({ error: 'not authenticated' });
}

app.post('/api/login', (req, res) => {
  if (!authRequired) return res.json({ ok: true, noAuth: true });
  const given = String(req.body?.password || '');
  const a = Buffer.from(given);
  const b = Buffer.from(APP_PASSWORD);
  const ok = a.length === b.length && crypto.timingSafeEqual(a, b);
  if (!ok) return res.status(401).json({ error: 'wrong password' });
  res.cookie('h3s', sessionToken(), {
    httpOnly: true, signed: true, sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    maxAge: 30 * 24 * 3600 * 1000,
  });
  res.json({ ok: true });
});

app.get('/api/session', (req, res) => {
  res.json({
    authed: !authRequired || req.signedCookies?.h3s === sessionToken(),
    noAuth: !authRequired,
  });
});

app.post('/api/logout', (req, res) => { res.clearCookie('h3s'); res.json({ ok: true }); });

app.get('/api/status', requireAuth, async (req, res) => {
  const deadlineIso = POD_DEADLINE || null;
  const secondsLeft = deadlineIso
    ? Math.max(0, Math.round((new Date(deadlineIso).getTime() - Date.now()) / 1000))
    : null;
  const base = {
    online: false, comfy: null, gpu: null,
    queue: { running: 0, pending: 0 },
    models: { fl2va: false, ref2va: false },
    deadline: { iso: deadlineIso, secondsLeft },
    fps: FPS, usdPerHour: GPU_USD_PER_HOUR,
  };
  if (!COMFY_URL) return res.json({ ...base, error: 'COMFY_URL is not configured' });
  try {
    const [stats, queue, unets] = await Promise.all([
      comfyJson('/system_stats', {}, 15000),
      comfyJson('/queue', {}, 15000),
      comfyJson('/object_info/UNETLoader', {}, 20000),
    ]);
    const dev = (stats.devices || [])[0] || {};
    const names = unets?.UNETLoader?.input?.required?.unet_name?.[0] || [];
    res.json({
      ...base,
      online: true,
      comfy: { version: stats.system?.comfyui_version || 'unknown' },
      gpu: {
        name: dev.name || 'GPU',
        vramTotalMb: Math.round((dev.vram_total || 0) / 1048576),
        vramUsedMb: Math.round(((dev.vram_total || 0) - (dev.vram_free || 0)) / 1048576),
      },
      queue: {
        running: (queue.queue_running || []).length,
        pending: (queue.queue_pending || []).length,
      },
      models: { fl2va: names.includes(CKPT.fl2va), ref2va: names.includes(CKPT.ref2va) },
      wsConnected: wsAlive,
    });
  } catch (e) {
    res.json({ ...base, error: `pod unreachable: ${e.message}` });
  }
});

app.post('/api/upload', requireAuth, upload.single('file'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'no file' });
  const mime = req.file.mimetype || '';
  const kind = mime.startsWith('video') ? 'video' : mime.startsWith('audio') ? 'audio' : 'image';
  try {
    const fd = new FormData();
    const safe = `h3s_${Date.now()}_${(req.file.originalname || 'file').replace(/[^\w.-]/g, '_')}`;
    fd.append('image', new Blob([req.file.buffer], { type: mime || 'application/octet-stream' }), safe);
    fd.append('overwrite', 'true');
    const r = await comfyFetch('/upload/image', { method: 'POST', body: fd }, 180000);
    if (!r.ok) return res.status(502).json({ error: `upload failed (${r.status})` });
    const j = await r.json();
    res.json({ name: j.name, kind, previewUrl: `/api/asset/${encodeURIComponent(j.name)}` });
  } catch (e) {
    res.status(502).json({ error: `upload failed: ${e.message}` });
  }
});

app.get('/api/asset/:name', requireAuth, async (req, res) => {
  try {
    const q = new URLSearchParams({ filename: req.params.name, type: 'input', subfolder: '' });
    const r = await comfyFetch(`/view?${q}`, {}, 60000);
    if (!r.ok) return res.sendStatus(404);
    res.set('Content-Type', r.headers.get('content-type') || 'application/octet-stream');
    res.set('Cache-Control', 'private, max-age=3600');
    res.send(Buffer.from(await r.arrayBuffer()));
  } catch { res.sendStatus(502); }
});

app.post('/api/generate', requireAuth, async (req, res) => {
  const b = req.body || {};
  const prompt = String(b.prompt || '').trim();
  if (!prompt) return res.status(400).json({ error: 'prompt is empty' });

  const mode = ['t2v', 'i2v', 'flf2v', 'r2v'].includes(b.mode) ? b.mode : 't2v';
  const width = clampMultiple(b.width, 1344);
  const height = clampMultiple(b.height, 768);
  const length = snapFrames(b.durationSec ?? 5);
  const steps = Math.min(60, Math.max(1, Math.round(Number(b.steps) || 20)));
  const seed = Number.isFinite(Number(b.seed)) && b.seed !== null && b.seed !== ''
    ? Math.abs(Math.round(Number(b.seed)))
    : crypto.randomInt(1, 2 ** 31);

  try {
    const stats = await comfyJson('/system_stats', {}, 12000);
    if (!stats) throw new Error('no response');
  } catch (e) {
    return res.status(409).json({ error: `the GPU pod is offline (${e.message})` });
  }

  const job = {
    id: crypto.randomUUID().slice(0, 8),
    state: 'queued', mode, prompt, width, height, length, steps, seed,
    firstFrame: b.firstFrame || null,
    lastFrame: b.lastFrame || null,
    refImages: b.refImages || [], refVideos: b.refVideos || [], refAudios: b.refAudios || [],
    refImageSize: b.refImageSize === 'max' ? 'max' : 'match',
    createdAt: Date.now(), step: 0,
  };

  const { graph, saveNode, posterNode } = buildGraph(job);
  job.saveNode = saveNode;
  job.posterNode = posterNode;

  try {
    const r = await comfyFetch('/prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: graph, client_id: CLIENT_ID }),
    }, 60000);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      const detail = body?.error?.message || body?.error || `HTTP ${r.status}`;
      const nodeErrs = body?.node_errors ? ` ${JSON.stringify(body.node_errors).slice(0, 300)}` : '';
      return res.status(400).json({ error: `ComfyUI rejected the graph: ${detail}${nodeErrs}` });
    }
    job.promptId = body.prompt_id;
    promptToJob.set(job.promptId, job.id);
    pushJob(job);
    res.json({
      jobId: job.id, length, actualSeconds: +(length / FPS).toFixed(2), seed,
      estimateSeconds: estimateSeconds({ width, height, frames: length, steps }),
    });
  } catch (e) {
    res.status(502).json({ error: `could not queue the job: ${e.message}` });
  }
});

app.get('/api/jobs/:id/events', requireAuth, (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'unknown job' });
  res.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.flushHeaders?.();

  if (!listeners.has(job.id)) listeners.set(job.id, new Set());
  listeners.get(job.id).add(res);

  // Replay current state so a late subscriber is not left blank.
  const snap = publicJob(job);
  if (job.state === 'done') {
    res.write(`data: ${JSON.stringify({
      type: 'done', jobId: job.id, videoUrl: snap.videoUrl, posterUrl: snap.posterUrl,
      durationSec: snap.durationSec, width: job.width, height: job.height,
      seed: job.seed, elapsedSec: job.elapsedSec,
    })}\n\n`);
    return res.end();
  }
  if (job.state === 'error') {
    res.write(`data: ${JSON.stringify({ type: 'error', message: job.error })}\n\n`);
    return res.end();
  }
  res.write(`data: ${JSON.stringify(
    job.state === 'decoding' ? { type: 'decoding' }
      : job.step ? { type: 'progress', step: job.step, totalSteps: job.steps, pct: (job.step / job.steps) * 100, etaSeconds: job.etaSeconds ?? null }
        : { type: 'queued', queuePosition: 0 }
  )}\n\n`);

  const ping = setInterval(() => { try { res.write(': ping\n\n'); } catch {} }, 15000);
  req.on('close', () => {
    clearInterval(ping);
    listeners.get(job.id)?.delete(res);
  });
});

app.get('/api/jobs', requireAuth, (req, res) => {
  const limit = Math.min(60, Math.max(1, Number(req.query.limit) || 30));
  res.json({ jobs: order.slice(0, limit).map((id) => publicJob(jobs.get(id))).filter(Boolean) });
});

app.get('/api/jobs/:id', requireAuth, (req, res) => {
  const j = jobs.get(req.params.id);
  if (!j) return res.status(404).json({ error: 'unknown job' });
  res.json(publicJob(j));
});

app.delete('/api/jobs/:id', requireAuth, (req, res) => {
  const j = jobs.get(req.params.id);
  if (!j) return res.status(404).json({ error: 'unknown job' });
  if (j.cachedVideo) fsp.unlink(j.cachedVideo).catch(() => {});
  if (j.cachedPoster) fsp.unlink(j.cachedPoster).catch(() => {});
  jobs.delete(j.id);
  const i = order.indexOf(j.id);
  if (i >= 0) order.splice(i, 1);
  res.json({ ok: true });
});

app.post('/api/cancel/:id', requireAuth, async (req, res) => {
  const j = jobs.get(req.params.id);
  if (!j) return res.status(404).json({ error: 'unknown job' });
  try {
    await comfyFetch('/interrupt', { method: 'POST' }, 15000);
    failJob(j.id, 'cancelled');
    res.json({ ok: true });
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

/** Range-capable static send, so <video> can seek. */
function sendFileRange(req, res, file, type) {
  let stat;
  try { stat = fs.statSync(file); } catch { return res.sendStatus(404); }
  const range = req.headers.range;
  res.set('Content-Type', type);
  res.set('Accept-Ranges', 'bytes');
  res.set('Cache-Control', 'private, max-age=86400');
  if (!range) {
    res.set('Content-Length', String(stat.size));
    return fs.createReadStream(file).pipe(res);
  }
  const m = /bytes=(\d*)-(\d*)/.exec(range);
  const start = m && m[1] ? parseInt(m[1], 10) : 0;
  const end = m && m[2] ? parseInt(m[2], 10) : stat.size - 1;
  if (start >= stat.size) {
    res.status(416).set('Content-Range', `bytes */${stat.size}`);
    return res.end();
  }
  res.status(206);
  res.set('Content-Range', `bytes ${start}-${end}/${stat.size}`);
  res.set('Content-Length', String(end - start + 1));
  fs.createReadStream(file, { start, end }).pipe(res);
}

app.get('/api/video/:id', requireAuth, (req, res) => {
  const j = jobs.get(req.params.id);
  if (!j?.cachedVideo) return res.sendStatus(404);
  sendFileRange(req, res, j.cachedVideo, 'video/mp4');
});

app.get('/api/poster/:id', requireAuth, (req, res) => {
  const j = jobs.get(req.params.id);
  if (!j?.cachedPoster) return res.sendStatus(404);
  sendFileRange(req, res, j.cachedPoster, 'image/png');
});

app.get('/healthz', (req, res) => res.json({ ok: true, comfyConfigured: !!COMFY_URL }));

app.use(express.static(path.join(__dirname, 'public'), { maxAge: '5m' }));
app.get('*', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

function clampMultiple(v, dflt) {
  const n = Math.round(Number(v) || dflt);
  const snapped = Math.round(n / 32) * 32;
  return Math.min(2048, Math.max(256, snapped));
}

app.listen(PORT, () => {
  console.log(`H3 Studio on :${PORT}`);
  console.log(`  ComfyUI : ${COMFY_URL || '(not configured)'}`);
  console.log(`  auth    : ${authRequired ? 'password' : 'OPEN — set APP_PASSWORD'}`);
  console.log(`  deadline: ${POD_DEADLINE || '(none)'}`);
});
