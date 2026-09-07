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
import { planStory } from './lib/story.js';
import { concatVideos, extractChainFrame, ffmpegAvailable } from './lib/video.js';
import { llmConfigured } from './lib/llm.js';
import { planStoryboard, reviseKeyframePrompt } from './lib/storyboard.js';
import * as pod from './lib/pod.js';

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

/**
 * Encoder and VAEs are always the stock ones — a finetune replaces the DiT only.
 * The DiT filename is discovered from the pod rather than hardcoded, because the same
 * app has to drive both the stock Comfy-Org checkpoint and the TenStrip/10Eros-Max
 * finetune, and their filenames have nothing in common.
 */
const CKPT = {
  clip: 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
  videoVae: 'minimax_h3_video_vae_fp16.safetensors',
  audioVae: 'minimax_h3_audio_vae_fp32.safetensors',
};

/** Filled in by refreshCheckpoints() from /object_info/UNETLoader. */
let DIT = { fl2va: null, ref2va: null, available: [] };

/** Chroma1-HD, the image model used for storyboard keyframes. Discovered, like the DiT. */
let IMG = { unet: null, clip: null, vae: null, available: false };

/**
 * Classify the DiT checkpoints the pod actually has.
 *
 * "hybrid" files (10Eros-Max beta5) fold first/last-frame and reference conditioning
 * into one set of weights, so one file serves both modes. Everything else is matched on
 * the fl2va / ref2va marker in its name.
 */
export function classifyCheckpoints(names) {
  const list = (Array.isArray(names) ? names : []).filter((n) => typeof n === 'string');
  const h3 = list.filter((n) => /h3/i.test(n));
  const pool = h3.length ? h3 : list;
  const find = (re) => pool.find((n) => re.test(n)) || null;

  const hybrid = find(/hybrid/i);
  return {
    available: pool,
    // Prefer an explicit fl2va/ref2va file; fall back to a hybrid, which does both.
    fl2va: find(/fl2va/i) || hybrid || pool[0] || null,
    ref2va: find(/ref2va/i) || hybrid || null,
  };
}

/**
 * Chroma1-HD needs three files that are not part of the H3 stack: the DiT itself, a
 * T5-XXL text encoder, and the FLUX autoencoder. All three must be present or image
 * generation is simply reported as unavailable — a half-installed image model that
 * fails at sampling time wastes a pod minute per attempt.
 */
export function classifyImageModels(unets, clips, vaes) {
  const find = (list, re) => (Array.isArray(list) ? list : []).find((n) => re.test(n)) || null;
  const unet = find(unets, /chroma/i);
  const clip = find(clips, /t5xxl/i);
  const vae = find(vaes, /^ae\.safetensors$|flux/i);
  return { unet, clip, vae, available: !!(unet && clip && vae) };
}

async function refreshCheckpoints() {
  try {
    const [u, c, v] = await Promise.all([
      comfyJson('/object_info/UNETLoader', {}, 20000),
      comfyJson('/object_info/CLIPLoader', {}, 20000).catch(() => null),
      comfyJson('/object_info/VAELoader', {}, 20000).catch(() => null),
    ]);
    const names = u?.UNETLoader?.input?.required?.unet_name?.[0] || [];
    DIT = classifyCheckpoints(names);
    IMG = classifyImageModels(
      names,
      c?.CLIPLoader?.input?.required?.clip_name?.[0] || [],
      v?.VAELoader?.input?.required?.vae_name?.[0] || [],
    );
  } catch { /* pod offline; /api/status reports that separately */ }
  return DIT;
}

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

  const unetName = (isRef ? DIT.ref2va : DIT.fl2va) || DIT.available[0];
  if (!unetName) throw new Error('no MiniMax H3 checkpoint is loaded on the pod');
  const unet = add('UNETLoader', { unet_name: unetName, weight_dtype: 'default' });
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

  // LoRAs patch the DiT only. Chain them, and feed the result to BOTH MODEL
  // consumers — patching the guider but not the scheduler would compute the
  // sigma schedule from different weights than the denoiser uses.
  let modelSrc = [unet, 0];
  for (const l of (job.loras || [])) {
    modelSrc = [add('LoraLoaderModelOnly', {
      model: modelSrc, lora_name: l.name, strength_model: l.strength,
    }), 0];
  }

  const noise = add('RandomNoise', { noise_seed: seed });
  const guider = add('BasicGuider', { model: modelSrc, conditioning: [cond, 0] });
  // The released checkpoints are CFG-distilled -> BasicGuider, not CFGGuider.
  const sampler = add('KSamplerSelect', { sampler_name: 'res_multistep' });
  const sigmas = add('BasicScheduler', {
    model: modelSrc, scheduler: 'simple', steps, denoise: 1.0,
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

/**
 * Chroma1-HD text-to-image graph, for storyboard keyframes.
 *
 * Unlike the H3 checkpoints, Chroma is NOT CFG-distilled: it takes a real negative
 * prompt and a CFG scale, so this uses an ordinary KSampler rather than the
 * BasicGuider/SamplerCustomAdvanced chain the video graph needs. The latent is
 * EmptySD3LatentImage because Chroma inherits FLUX's 16-channel latent space.
 */
function buildChromaGraph(job) {
  const { prompt, negative, width, height, steps, cfg, seed } = job;
  if (!IMG.available) throw new Error('Chroma1-HD is not installed on this pod');
  const g = {};
  const add = (class_type, inputs) => {
    const id = nid();
    g[id] = { class_type, inputs };
    return id;
  };

  const unet = add('UNETLoader', { unet_name: IMG.unet, weight_dtype: 'default' });
  const clip = add('CLIPLoader', { clip_name: IMG.clip, type: 'chroma', device: 'default' });
  const vae = add('VAELoader', { vae_name: IMG.vae });

  const pos = add('CLIPTextEncode', { clip: [clip, 0], text: prompt });
  const neg = add('CLIPTextEncode', { clip: [clip, 0], text: negative || '' });
  const latent = add('EmptySD3LatentImage', { width, height, batch_size: 1 });

  const sampled = add('KSampler', {
    model: [unet, 0], positive: [pos, 0], negative: [neg, 0], latent_image: [latent, 0],
    seed, steps, cfg, sampler_name: 'euler', scheduler: 'beta', denoise: 1.0,
  });
  const img = add('VAEDecode', { samples: [sampled, 0], vae: [vae, 0] });
  const save = add('SaveImage', { images: [img, 0], filename_prefix: `h3keyframe/${job.id}` });
  return { graph: g, saveNode: save };
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
    loras: j.loras || [],
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
  // Ten years. The brief is "type it once and never again"; a 30-day cookie means
  // hunting for the password every month, which in practice means writing it down.
  res.cookie('h3s', sessionToken(), {
    httpOnly: true, signed: true, sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    maxAge: 10 * 365 * 24 * 3600 * 1000,
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
    models: { fl2va: false, ref2va: false, dit: null, available: [] },
    story: { llm: llmConfigured(), ffmpeg: ffmpegReady },
    image: { ...IMG },
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
    DIT = classifyCheckpoints(names);
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
      models: {
        fl2va: !!DIT.fl2va, ref2va: !!DIT.ref2va,
        dit: DIT.fl2va, available: DIT.available,
        finetune: /eros/i.test(DIT.fl2va || '') ? '10Eros-Max' : 'stock MiniMax H3',
      },
      story: { llm: llmConfigured(), ffmpeg: ffmpegReady },
      image: { ...IMG },
      wsConnected: wsAlive,
    });
  } catch (e) {
    res.json({ ...base, error: `pod unreachable: ${e.message}` });
  }
});

app.get('/api/loras', requireAuth, async (req, res) => {
  try {
    const oi = await comfyJson('/object_info/LoraLoaderModelOnly', {}, 20000);
    const names = oi?.LoraLoaderModelOnly?.input?.required?.lora_name?.[0] || [];
    res.json({ loras: (Array.isArray(names) ? names : []).map((n) => ({ name: n })) });
  } catch (e) {
    res.json({ loras: [], error: e.message });
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

/**
 * Build, queue and register one generation job.
 *
 * Split out of the /api/generate route so story mode can drive the identical path —
 * a story segment must be exactly the same kind of job as a hand-made clip, or the two
 * drift apart the first time either is changed.
 *
 * Returns the job; throws with a readable message when ComfyUI rejects the graph.
 */
async function submitJob(b) {
  const prompt = String(b.prompt || '').trim();
  if (!prompt) throw new Error('prompt is empty');

  const mode = ['t2v', 'i2v', 'flf2v', 'r2v'].includes(b.mode) ? b.mode : 't2v';
  const width = clampMultiple(b.width, 1344);
  const height = clampMultiple(b.height, 768);
  const length = b.length ? Number(b.length) : snapFrames(b.durationSec ?? 5);
  const steps = Math.min(60, Math.max(1, Math.round(Number(b.steps) || 20)));
  const seed = Number.isFinite(Number(b.seed)) && b.seed !== null && b.seed !== ''
    ? Math.abs(Math.round(Number(b.seed)))
    : crypto.randomInt(1, 2 ** 31);

  if (!DIT.fl2va) await refreshCheckpoints();

  const job = {
    id: crypto.randomUUID().slice(0, 8),
    state: 'queued', mode, prompt, width, height, length, steps, seed,
    firstFrame: b.firstFrame || null,
    lastFrame: b.lastFrame || null,
    refImages: b.refImages || [], refVideos: b.refVideos || [], refAudios: b.refAudios || [],
    refImageSize: b.refImageSize === 'max' ? 'max' : 'match',
    loras: (Array.isArray(b.loras) ? b.loras : [])
      .filter((l) => l && typeof l.name === 'string' && l.name)
      .slice(0, 6)
      .map((l) => ({
        name: l.name,
        strength: Math.max(-4, Math.min(4, Number(l.strength ?? 1) || 0)),
      })),
    storyId: b.storyId || null,
    createdAt: Date.now(), step: 0,
  };

  const { graph, saveNode, posterNode } = buildGraph(job);
  job.saveNode = saveNode;
  job.posterNode = posterNode;

  const r = await comfyFetch('/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: graph, client_id: CLIENT_ID }),
  }, 60000);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = body?.error?.message || body?.error || `HTTP ${r.status}`;
    const nodeErrs = body?.node_errors ? ` ${JSON.stringify(body.node_errors).slice(0, 300)}` : '';
    throw new Error(`ComfyUI rejected the graph: ${detail}${nodeErrs}`);
  }
  job.promptId = body.prompt_id;
  promptToJob.set(job.promptId, job.id);
  pushJob(job);
  return job;
}

/** Resolve when a job reaches a terminal state. Used by the story renderer. */
function waitForJob(jobId, { signal } = {}) {
  return new Promise((resolve, reject) => {
    const tick = setInterval(() => {
      if (signal?.aborted) {
        clearInterval(tick);
        return reject(new Error('cancelled'));
      }
      const j = jobs.get(jobId);
      if (!j) { clearInterval(tick); return reject(new Error('job disappeared')); }
      if (j.state === 'done') { clearInterval(tick); return resolve(j); }
      if (j.state === 'error') { clearInterval(tick); return reject(new Error(j.error || 'generation failed')); }
    }, 1500);
    tick.unref?.();
  });
}

app.post('/api/generate', requireAuth, async (req, res) => {
  const b = req.body || {};
  try {
    const stats = await comfyJson('/system_stats', {}, 12000);
    if (!stats) throw new Error('no response');
  } catch (e) {
    return res.status(409).json({ error: `the GPU pod is offline (${e.message})` });
  }
  try {
    const job = await submitJob(b);
    res.json({
      jobId: job.id,
      length: job.length,
      actualSeconds: +(job.length / FPS).toFixed(2),
      seed: job.seed,
      estimateSeconds: estimateSeconds({
        width: job.width, height: job.height, frames: job.length, steps: job.steps,
      }),
    });
  } catch (e) {
    const offline = /rejected the graph/.test(e.message) ? 400 : 502;
    res.status(offline).json({ error: e.message });
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

// ------------------------------------------------------------- story mode

/**
 * A story is a plan (from the LLM) plus one generation job per segment, chained so that
 * each segment starts on the frame the previous one ended on, and finally concatenated.
 */
const stories = new Map();
const storyOrder = [];
const storyListeners = new Map();
let ffmpegReady = false;

function emitStory(id, payload) {
  const set = storyListeners.get(id);
  if (!set) return;
  const line = `data: ${JSON.stringify(payload)}\n\n`;
  for (const res of set) { try { res.write(line); } catch { /* gone */ } }
  if (payload.type === 'done' || payload.type === 'error') {
    for (const res of set) { try { res.end(); } catch {} }
    storyListeners.delete(id);
  }
}

function publicStory(st) {
  if (!st) return null;
  return {
    storyId: st.id, kind: st.kind || 'story',
    state: st.state, stage: st.stage, title: st.plan?.title,
    logline: st.plan?.logline, synopsis: st.plan?.synopsis,
    characters: st.plan?.characters || [],
    segmentCount: st.plan?.prompts?.length || 0,
    secondsPerSegment: st.plan?.secondsPerSegment,
    totalSeconds: st.plan?.totalSeconds,
    width: st.width, height: st.height, steps: st.steps,
    segments: (st.segments || []).map((sg) => ({
      index: sg.index, title: sg.title, state: sg.state, jobId: sg.jobId,
      prompt: sg.prompt, error: sg.error || undefined,
      videoUrl: sg.jobId && jobs.get(sg.jobId)?.state === 'done' ? `/api/video/${sg.jobId}` : undefined,
    })),
    createdAt: st.createdAt,
    elapsedSec: st.elapsedSec ?? null,
    error: st.error || undefined,
    videoUrl: st.state === 'done' ? `/api/story/${st.id}/video` : undefined,
    concatMethod: st.concatMethod,
  };
}

/**
 * Render every segment in order, chaining frames between them.
 *
 * Strictly sequential on purpose. One pod means one GPU, so parallel submission would
 * only fill ComfyUI's queue — and more importantly segment N+1 cannot start before
 * segment N exists, because it begins on N's final frame.
 */
async function renderStory(st) {
  st.state = 'rendering';
  st.startedAt = Date.now();
  const files = [];
  let chainFrameName = st.startFrame || null;

  try {
    for (let i = 0; i < st.plan.prompts.length; i++) {
      if (st.abort.signal.aborted) throw new Error('cancelled');
      const seg = st.segments[i];
      const p = st.plan.prompts[i];

      st.stage = `segment ${i + 1} of ${st.plan.prompts.length}`;
      seg.state = 'running';
      emitStory(st.id, { type: 'segment', index: seg.index, state: 'running', stage: st.stage });

      // Storyboard mode pins BOTH ends of every shot to approved keyframes, so nothing
      // is chained and nothing is invented at either boundary. Story mode pins only the
      // opening frame, carried over from the previous shot.
      let firstFrame, lastFrame;
      if (st.kind === 'storyboard') {
        firstFrame = st.keyframeNames[i];
        lastFrame = st.keyframeNames[i + 1] || null;
      } else {
        const isLast = i === st.plan.prompts.length - 1;
        firstFrame = chainFrameName;
        lastFrame = isLast ? (st.endFrame || null) : null;
      }
      const mode = firstFrame ? (lastFrame ? 'flf2v' : 'i2v') : (lastFrame ? 'flf2v' : 't2v');

      const job = await submitJob({
        prompt: p.prompt,
        mode,
        width: st.width, height: st.height,
        length: st.plan.frames,
        steps: st.steps,
        seed: st.seed === null ? null : st.seed + i,
        firstFrame,
        lastFrame,
        loras: st.loras,
        storyId: st.id,
      });
      seg.jobId = job.id;
      emitStory(st.id, { type: 'segment', index: seg.index, state: 'running', jobId: job.id });

      const done = await waitForJob(job.id, { signal: st.abort.signal });
      seg.state = 'done';
      files.push(done.cachedVideo);
      emitStory(st.id, {
        type: 'segment', index: seg.index, state: 'done', jobId: job.id,
        videoUrl: `/api/video/${job.id}`,
      });

      // Hand the next segment the frame this one ended on. Storyboard mode already knows
      // what every shot starts on, so it skips this entirely.
      if (st.kind !== 'storyboard' && i < st.plan.prompts.length - 1) {
        const framePath = path.join(CACHE_DIR, `${st.id}-chain-${i}.jpg`);
        await extractChainFrame(done.cachedVideo, framePath, { offsetSec: st.chainOffsetSec });
        chainFrameName = await uploadToComfy(framePath, `chain_${st.id}_${i}.jpg`);
      }
    }

    st.stage = 'stitching';
    emitStory(st.id, { type: 'stage', stage: 'stitching' });
    const out = path.join(CACHE_DIR, `${st.id}-final.mp4`);
    const r = await concatVideos(files, out, { workDir: CACHE_DIR });
    st.finalVideo = out;
    st.concatMethod = r.method;
    st.state = 'done';
    st.stage = 'done';
    st.elapsedSec = Math.round((Date.now() - st.startedAt) / 1000);
    emitStory(st.id, {
      type: 'done', storyId: st.id, videoUrl: `/api/story/${st.id}/video`,
      durationSec: r.durationSec, elapsedSec: st.elapsedSec, method: r.method,
    });
  } catch (e) {
    st.state = 'error';
    st.error = e.message;
    const cur = st.segments.find((x) => x.state === 'running');
    if (cur) { cur.state = 'error'; cur.error = e.message; }
    emitStory(st.id, { type: 'error', message: e.message });
  }
}

/** Push a local file into ComfyUI's input/ directory and return the name it got. */
async function uploadToComfy(filePath, name, type = 'image/jpeg') {
  const buf = await fsp.readFile(filePath);
  const fd = new FormData();
  fd.append('image', new Blob([buf], { type }), name);
  fd.append('overwrite', 'true');
  const r = await comfyFetch('/upload/image', { method: 'POST', body: fd }, 180000);
  if (!r.ok) throw new Error(`chain-frame upload failed (${r.status})`);
  return (await r.json()).name;
}

app.post('/api/story/plan', requireAuth, async (req, res) => {
  if (!llmConfigured()) {
    return res.status(503).json({ error: 'ABLITERATION_API_KEY is not configured' });
  }
  try {
    const plan = await planStory(req.body || {});
    res.json({ plan });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

app.post('/api/story/render', requireAuth, async (req, res) => {
  const b = req.body || {};
  const plan = b.plan;
  if (!plan?.prompts?.length) return res.status(400).json({ error: 'no plan supplied' });
  if (!ffmpegReady) {
    return res.status(503).json({ error: 'ffmpeg is not available in this container' });
  }
  try {
    await comfyJson('/system_stats', {}, 12000);
  } catch (e) {
    return res.status(409).json({ error: `the GPU pod is offline (${e.message})` });
  }

  // Edits made in the UI's prompt boxes must win over what the LLM originally wrote.
  if (Array.isArray(b.prompts)) {
    plan.prompts = plan.prompts.map((p, i) => (
      typeof b.prompts[i] === 'string' && b.prompts[i].trim()
        ? { ...p, prompt: b.prompts[i].trim() }
        : p));
  }

  const st = {
    id: crypto.randomUUID().slice(0, 8),
    plan,
    state: 'queued', stage: 'queued',
    width: clampMultiple(b.width, 1344),
    height: clampMultiple(b.height, 768),
    steps: Math.min(60, Math.max(1, Math.round(Number(b.steps) || 8))),
    seed: b.seed === '' || b.seed === null || b.seed === undefined
      ? null : Math.abs(Math.round(Number(b.seed))) || null,
    startFrame: b.startFrame || null,
    endFrame: b.endFrame || null,
    chainOffsetSec: Math.min(1, Math.max(0, Number(b.chainOffsetSec ?? 0.12))),
    loras: Array.isArray(b.loras) ? b.loras : [],
    segments: plan.prompts.map((p) => ({
      index: p.index, title: p.title, prompt: p.prompt, state: 'pending', jobId: null,
    })),
    createdAt: Date.now(),
    abort: new AbortController(),
  };

  stories.set(st.id, st);
  storyOrder.unshift(st.id);
  while (storyOrder.length > 20) {
    const gone = storyOrder.pop();
    const old = stories.get(gone);
    if (old?.finalVideo) fsp.unlink(old.finalVideo).catch(() => {});
    stories.delete(gone);
  }

  renderStory(st).catch((e) => {
    st.state = 'error';
    st.error = e.message;
  });

  res.json({ storyId: st.id, segmentCount: st.segments.length });
});

app.get('/api/story', requireAuth, (req, res) => {
  res.json({ stories: storyOrder.map((id) => publicStory(stories.get(id))).filter(Boolean) });
});

app.get('/api/story/:id', requireAuth, (req, res) => {
  const st = stories.get(req.params.id);
  if (!st) return res.status(404).json({ error: 'unknown story' });
  res.json(publicStory(st));
});

app.get('/api/story/:id/events', requireAuth, (req, res) => {
  const st = stories.get(req.params.id);
  if (!st) return res.status(404).json({ error: 'unknown story' });
  res.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.flushHeaders?.();
  if (!storyListeners.has(st.id)) storyListeners.set(st.id, new Set());
  storyListeners.get(st.id).add(res);

  res.write(`data: ${JSON.stringify({ type: 'snapshot', story: publicStory(st) })}\n\n`);
  if (st.state === 'done' || st.state === 'error') return res.end();

  const ping = setInterval(() => { try { res.write(': ping\n\n'); } catch {} }, 15000);
  req.on('close', () => {
    clearInterval(ping);
    storyListeners.get(st.id)?.delete(res);
  });
});

app.post('/api/story/:id/cancel', requireAuth, async (req, res) => {
  const st = stories.get(req.params.id);
  if (!st) return res.status(404).json({ error: 'unknown story' });
  st.abort.abort();
  try { await comfyFetch('/interrupt', { method: 'POST' }, 15000); } catch { /* best effort */ }
  res.json({ ok: true });
});

app.get('/api/story/:id/video', requireAuth, (req, res) => {
  const st = stories.get(req.params.id);
  if (!st?.finalVideo) return res.sendStatus(404);
  sendFileRange(req, res, st.finalVideo, 'video/mp4');
});

// -------------------------------------------------------------- pod control

/**
 * The reaper on Sandy asks this before falling back to the pod's own H3_DEADLINE env and
 * then to its blanket one-hour cap. Deliberately readable with a shared token rather than
 * the UI session cookie, because the caller is a cron job, not a browser.
 */
app.get('/api/pod/deadlines', (req, res) => {
  const want = process.env.REAPER_TOKEN || '';
  const got = req.get('X-Reaper-Token') || req.query.token || '';
  if (!want || got !== want) return res.status(401).json({ error: 'bad token' });
  res.json({ deadlines: pod.deadlines(), now: Math.round(Date.now() / 1000) });
});

app.get('/api/pod/status', requireAuth, async (req, res) => {
  try { await pod.refresh(); } catch { /* status still reports what we know */ }
  res.json(pod.status());
});

app.post('/api/pod/start', requireAuth, async (req, res) => {
  const hours = Number(req.body?.hours);
  if (!Number.isFinite(hours) || hours <= 0) {
    return res.status(400).json({ error: 'hours is required' });
  }
  if (hours > pod.MAX_HOURS) {
    return res.status(400).json({ error: `the cap is ${pod.MAX_HOURS} hours` });
  }
  try {
    res.json(await pod.start({ hours, chroma: req.body?.chroma !== false }));
  } catch (e) {
    res.status(409).json({ error: e.message });
  }
});

app.post('/api/pod/stop', requireAuth, async (req, res) => {
  try {
    res.json(await pod.stop('stopped from the web app'));
  } catch (e) {
    res.status(409).json({ error: e.message });
  }
});

app.post('/api/pod/extend', requireAuth, (req, res) => {
  try {
    res.json(pod.extend(Number(req.body?.hours)));
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

app.get('/api/pod/events', requireAuth, (req, res) => {
  res.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.flushHeaders?.();
  res.write(`data: ${JSON.stringify({ type: 'snapshot', ...pod.status() })}\n\n`);
  const off = pod.subscribe(res);
  const ping = setInterval(() => { try { res.write(': ping\n\n'); } catch {} }, 15000);
  req.on('close', () => { clearInterval(ping); off(); });
});

// ------------------------------------------------------------ image + storyboard

const images = new Map();   // imageId -> { file, prompt, seed, ... }
const imageOrder = [];

/** Generate one Chroma keyframe and cache the PNG locally. Resolves when it is done. */
async function generateImage(spec) {
  if (!IMG.available) await refreshCheckpoints();
  const id = crypto.randomUUID().slice(0, 8);
  const job = {
    id,
    prompt: String(spec.prompt || '').trim(),
    negative: String(spec.negative || ''),
    width: clampMultiple(spec.width, 1344),
    height: clampMultiple(spec.height, 768),
    steps: Math.min(60, Math.max(4, Math.round(Number(spec.steps) || 26))),
    cfg: Math.min(12, Math.max(1, Number(spec.cfg) || 4.0)),
    seed: Number.isFinite(Number(spec.seed)) && spec.seed !== null && spec.seed !== ''
      ? Math.abs(Math.round(Number(spec.seed)))
      : crypto.randomInt(1, 2 ** 31),
  };
  if (!job.prompt) throw new Error('prompt is empty');

  const { graph, saveNode } = buildChromaGraph(job);
  const r = await comfyFetch('/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: graph, client_id: CLIENT_ID }),
  }, 60000);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = body?.error?.message || body?.error || `HTTP ${r.status}`;
    throw new Error(`ComfyUI rejected the image graph: ${detail}`);
  }
  const promptId = body.prompt_id;

  // Images are quick (~20 s) and there is no per-step feedback worth streaming, so this
  // polls history rather than joining the websocket fan-out the video jobs use.
  const deadline = Date.now() + 15 * 60 * 1000;
  while (Date.now() < deadline) {
    await new Promise((r2) => setTimeout(r2, 2000));
    let hist;
    try { hist = await comfyJson(`/history/${promptId}`, {}, 20000); } catch { continue; }
    const entry = hist?.[promptId];
    if (!entry) continue;
    const st = entry.status || {};
    if (st.status_str === 'error') {
      const m = (st.messages || []).find((x) => x[0] === 'execution_error');
      throw new Error(m ? JSON.stringify(m[1]).slice(0, 300) : 'image generation failed');
    }
    if (!st.completed) continue;
    const out = (entry.outputs || {})[saveNode] || {};
    const ref = (out.images || [])[0];
    if (!ref) throw new Error('no image in ComfyUI output');
    const file = await cacheFile(ref, `${id}.png`);
    const rec = { id, file, ...job, createdAt: Date.now() };
    images.set(id, rec);
    imageOrder.unshift(id);
    while (imageOrder.length > 200) {
      const gone = imageOrder.pop();
      const old = images.get(gone);
      if (old?.file) fsp.unlink(old.file).catch(() => {});
      images.delete(gone);
    }
    await pruneCache();
    return rec;
  }
  throw new Error('image generation timed out');
}

app.post('/api/image/generate', requireAuth, async (req, res) => {
  try {
    const rec = await generateImage(req.body || {});
    res.json({ imageId: rec.id, url: `/api/image/${rec.id}`, seed: rec.seed,
               width: rec.width, height: rec.height });
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

app.get('/api/image/:id', requireAuth, (req, res) => {
  const rec = images.get(req.params.id);
  if (!rec?.file) return res.sendStatus(404);
  sendFileRange(req, res, rec.file, 'image/png');
});

/**
 * Push a generated image into ComfyUI's input/ directory.
 *
 * Necessary because LoadImage resolves by filename against input/ and cannot read from
 * output/, where SaveImage put it. So a keyframe has to make a round trip through this
 * app to become usable as a first_frame.
 */
app.post('/api/image/:id/as-input', requireAuth, async (req, res) => {
  const rec = images.get(req.params.id);
  if (!rec?.file) return res.status(404).json({ error: 'unknown image' });
  try {
    const name = await uploadToComfy(rec.file, `kf_${rec.id}.png`, 'image/png');
    rec.comfyName = name;
    res.json({ name });
  } catch (e) {
    res.status(502).json({ error: e.message });
  }
});

app.post('/api/storyboard/plan', requireAuth, async (req, res) => {
  if (!llmConfigured()) return res.status(503).json({ error: 'ABLITERATION_API_KEY is not configured' });
  try {
    res.json({ board: await planStoryboard(req.body || {}) });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

app.post('/api/storyboard/revise-prompt', requireAuth, async (req, res) => {
  try {
    res.json({ prompt: await reviseKeyframePrompt(req.body || {}) });
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

/**
 * Render a storyboard: every shot is a first+last frame interpolation between two
 * approved keyframes, so both ends are pinned and the model only invents the motion.
 */
app.post('/api/storyboard/render', requireAuth, async (req, res) => {
  const b = req.body || {};
  const board = b.board;
  const frames = b.keyframeNames;   // [comfyui input filename per keyframe, in order]
  if (!board?.shots?.length) return res.status(400).json({ error: 'no storyboard supplied' });
  if (!Array.isArray(frames) || frames.length !== board.keyframes.length) {
    return res.status(400).json({ error: 'every keyframe must be generated before rendering' });
  }
  if (frames.some((f) => !f)) return res.status(400).json({ error: 'a keyframe is still missing' });
  if (!ffmpegReady) return res.status(503).json({ error: 'ffmpeg is not available in this container' });
  try { await comfyJson('/system_stats', {}, 12000); }
  catch (e) { return res.status(409).json({ error: `the GPU pod is offline (${e.message})` }); }

  const plan = {
    ...board,
    prompts: board.shots.map((sh, i) => ({
      index: i + 1,
      title: sh.title,
      prompt: (Array.isArray(b.prompts) && typeof b.prompts[i] === 'string' && b.prompts[i].trim())
        ? b.prompts[i].trim() : sh.prompt,
    })),
    frames: snapFrames(board.secondsPerSegment ?? 5),
    secondsPerSegment: board.secondsPerSegment ?? 5,
  };

  const st = {
    id: crypto.randomUUID().slice(0, 8),
    plan,
    kind: 'storyboard',
    keyframeNames: frames,
    state: 'queued', stage: 'queued',
    width: clampMultiple(b.width, 1344),
    height: clampMultiple(b.height, 768),
    steps: Math.min(60, Math.max(1, Math.round(Number(b.steps) || 6))),
    seed: b.seed === '' || b.seed === null || b.seed === undefined
      ? null : Math.abs(Math.round(Number(b.seed))) || null,
    chainOffsetSec: 0.12,
    loras: Array.isArray(b.loras) ? b.loras : [],
    segments: plan.prompts.map((p) => ({
      index: p.index, title: p.title, prompt: p.prompt, state: 'pending', jobId: null,
    })),
    createdAt: Date.now(),
    abort: new AbortController(),
  };

  stories.set(st.id, st);
  storyOrder.unshift(st.id);
  renderStory(st).catch((e) => { st.state = 'error'; st.error = e.message; });
  res.json({ storyId: st.id, segmentCount: st.segments.length });
});

app.get('/healthz', (req, res) => res.json({ ok: true, comfyConfigured: !!COMFY_URL }));

app.use(express.static(path.join(__dirname, 'public'), { maxAge: '5m' }));
app.get('*', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

function clampMultiple(v, dflt) {
  const n = Math.round(Number(v) || dflt);
  const snapped = Math.round(n / 32) * 32;
  return Math.min(2048, Math.max(256, snapped));
}

pod.init().catch((e) => console.warn('  pod control init failed:', e.message));

ffmpegAvailable().then((ok) => {
  ffmpegReady = ok;
  if (!ok) console.warn('  ffmpeg  : MISSING — story mode is disabled');
});

app.listen(PORT, () => {
  console.log(`H3 Studio on :${PORT}`);
  console.log(`  ComfyUI : ${COMFY_URL || '(not configured)'}`);
  console.log(`  auth    : ${authRequired ? 'password' : 'OPEN — set APP_PASSWORD'}`);
  console.log(`  deadline: ${POD_DEADLINE || '(none)'}`);
  console.log(`  pod ctrl: ${pod.podControlConfigured() ? 'enabled' : 'disabled (needs RUNPOD_API_KEY + POD_SSH_*)'}`);
});
