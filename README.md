# H3 Studio

A web UI for generating video with **MiniMax H3** on a self-hosted GPU.

H3 is an open-weights omni-modal model: it produces 4–15 s of video *and a natively
synchronised stereo soundtrack* — dialogue, ambience, music, foley — from one prompt.
There is no separate TTS anywhere in this stack.

The heavy lifting runs in ComfyUI on a RunPod **RTX PRO 6000 Blackwell** (96 GB).
This app is the front door: it builds the ComfyUI graphs, streams progress, and
caches results so the gallery survives the GPU pod being torn down.

![modes](https://img.shields.io/badge/modes-T2V%20%7C%20I2V%20%7C%20First%2BLast%20%7C%20Reference-blue)

## Modes

| Mode | Node | Inputs |
|---|---|---|
| Text → Video | `MiniMaxH3ImageToVideo` | prompt only |
| Image → Video | `MiniMaxH3ImageToVideo` | prompt + first frame |
| First + Last Frame | `MiniMaxH3ImageToVideo` | prompt + either/both keyframes |
| Reference → Video | `MiniMaxH3ReferenceToVideo` | prompt + ≤9 images, ≤3 videos, ≤3 audio clips |

Reference mode needs the `ref2va` checkpoint on the pod; the UI disables the tab when
it is not present. Refer to uploads from the prompt as `<Picture 1>`, `<Picture 2>`, …

## Architecture

```
browser ──HTTPS──> Render (this app, free tier)
                        │
                        ├── REST + SSE to the browser
                        └── HTTPS + WSS ──> https://<podId>-8188.proxy.runpod.net
                                             ComfyUI 0.30 on RunPod
                                             MiniMax H3 (int8 + nvfp4)
```

RunPod publishes any http port declared on a pod at `https://<podId>-<port>.proxy.runpod.net`,
so no tunnel or VPN is involved. The pod URL is never exposed to the browser — every
call is proxied, and the whole app sits behind one shared password.

Three things justify a server rather than calling ComfyUI from the page:

1. **One websocket, many viewers.** ComfyUI reports sampler progress over a single WS;
   the server holds it and fans out per-job SSE.
2. **Graph construction.** The templates ComfyUI ships are UI-format workflows wrapping a
   subgraph and cannot be POSTed to `/prompt`. The server emits the flattened API graph.
3. **Outliving the pod.** Finished mp4s and posters are copied to local disk, so history
   still plays after the GPU is gone.

## Configuration

| Variable | Required | Meaning |
|---|---|---|
| `COMFY_URL` | yes | e.g. `https://abc123-8188.proxy.runpod.net` |
| `APP_PASSWORD` | strongly advised | shared password; **unset means the app is open to anyone with the URL** |
| `SESSION_SECRET` | no | signs the session cookie; random per boot if unset (logs everyone out on restart) |
| `POD_DEADLINE` | no | ISO8601; drives the "pod auto-terminates in …" countdown |
| `GPU_USD_PER_HOUR` | no | default `2.09`, used for the cost estimate |
| `CACHE_MAX_BYTES` | no | default 250 MB of cached videos |

## Running locally

```bash
npm install
COMFY_URL=https://<podId>-8188.proxy.runpod.net APP_PASSWORD=dev npm start
# http://localhost:3000
```

## Pointing it at a new pod

GPU pods here are deliberately short-lived — they are killed on a timer so an idle
GPU cannot quietly bill for days. When the pod dies the app stays up and reports
"pod offline". To bring it back: launch a new pod (see the runbook in
`../docs/01-pod-setup.md`), then update `COMFY_URL` and `POD_DEADLINE` on the Render
service. No redeploy needed beyond the automatic restart.

## Notes on the model that shape the UI

- **Frame grid.** The video VAE only accepts frame counts where `length % 17 == 5`
  (5, 22, 39, … 243). The duration slider snaps to that grid and shows the real value.
- **No CFG.** The released checkpoints are CFG-distilled, so there is no guidance
  scale to expose — `BasicGuider`, not `CFGGuider`.
- **Cost is superlinear in length.** Attention is quadratic in sequence length: 124
  frames at 1344×768 takes ~370 s, 243 frames takes ~917 s. The estimate in the UI
  models this and is still only an estimate.
- **768p is the ceiling here.** H3's 2K path (`H3-Regenerate-2K`) is not open-sourced;
  it exists only behind MiniMax's hosted API.

## Licence / usage

The MiniMax H3 weights are under the MiniMax H3 Community License, which carries use
restrictions. The hosted MiniMax API applies content moderation; local weights do not.
