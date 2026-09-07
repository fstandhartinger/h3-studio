#!/bin/bash
# Runs ON the pod. Installs ComfyUI, MiniMax H3 / 10Eros-Max, and Chroma1-HD.
# Uploaded and started by the web app (lib/pod.js) over ssh.
#
# Progress is reported by writing single-word phases to /workspace/phase, which the app
# polls. Anything more structured would be nicer, but this survives a dropped ssh
# connection, which a streamed stdout does not.
exec >> /workspace/setup.log 2>&1
set -x
export PIP_ROOT_USER_ACTION=ignore
phase() { echo "$1" > /workspace/phase; date -u +"PHASE $1 %H:%M:%S"; }

phase apt
apt-get update -qq && apt-get install -y -qq git ffmpeg &
APT=$!

mkdir -p /workspace/models/{diffusion_models,text_encoders,vae}

phase weights
HF=https://huggingface.co
# Hugging Face throttles per TCP connection, so every file gets its own pool of parallel
# range requests. Connection counts are roughly proportional to file size.
python3 /root/pdl.py "$HF/TenStrip/10Eros-Max/resolve/main/10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors" \
  /workspace/models/diffusion_models/10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors 24 & D1=$!
python3 /root/pdl.py "$HF/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  /workspace/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors 16 & D2=$!
python3 /root/pdl.py "$HF/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" \
  /workspace/models/vae/minimax_h3_video_vae_fp16.safetensors 6 & D3=$!
python3 /root/pdl.py "$HF/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" \
  /workspace/models/vae/minimax_h3_audio_vae_fp32.safetensors 4 & D4=$!

if [ "${WANT_CHROMA:-1}" = "1" ]; then
  python3 /root/pdl.py "$HF/lodestones/Chroma1-HD/resolve/main/Chroma1-HD.safetensors" \
    /workspace/models/diffusion_models/Chroma1-HD.safetensors 24 & D5=$!
  python3 /root/pdl.py "$HF/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" \
    /workspace/models/text_encoders/t5xxl_fp16.safetensors 12 & D6=$!
  python3 /root/pdl.py "$HF/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors" \
    /workspace/models/vae/ae.safetensors 4 & D7=$!
fi

wait $APT
phase comfyui
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
# torchvision/torchaudio MUST be pinned, or pip quietly drags torch 2.9.1 forward.
python -m pip install --break-system-packages -q torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install --break-system-packages -q -r /workspace/ComfyUI/requirements.txt
python -m pip install --break-system-packages -q numpy opencv-python-headless imageio-ffmpeg av

phase weights
wait $D1 $D2 $D3 $D4 ${D5:-} ${D6:-} ${D7:-}

phase verify
cd /workspace/ComfyUI/models
mkdir -p diffusion_models text_encoders vae loras
ln -sf /workspace/models/diffusion_models/*.safetensors diffusion_models/
ln -sf /workspace/models/text_encoders/*.safetensors    text_encoders/
ln -sf /workspace/models/vae/*.safetensors              vae/

# A truncated checkpoint loads far enough to waste a whole pod before it fails, so the
# safetensors header is checked against the file size before ComfyUI ever sees it.
python3 - <<'PYV'
import json, os, struct, sys
bad = []
for root, _, files in os.walk('/workspace/models'):
    for f in sorted(files):
        if not f.endswith('.safetensors'):
            continue
        p = os.path.join(root, f)
        try:
            with open(p, 'rb') as fh:
                n = struct.unpack('<Q', fh.read(8))[0]
                h = json.loads(fh.read(n))
            need = 8 + n + max(v['data_offsets'][1] for k, v in h.items() if k != '__metadata__')
        except Exception as e:
            print('%-60s UNREADABLE %s' % (f, e)); bad.append(f); continue
        got = os.path.getsize(p)
        print('%-60s %s' % (f, 'OK' if need == got else 'TRUNCATED need=%d got=%d' % (need, got)))
        if need != got:
            bad.append(f)
sys.exit(1 if bad else 0)
PYV
if [ $? -ne 0 ]; then phase failed; exit 1; fi

phase starting
cd /workspace/ComfyUI
setsid nohup python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch \
  > /workspace/comfy.log 2>&1 < /dev/null &

for i in $(seq 1 60); do
  curl -sf --max-time 5 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && { phase ready; exit 0; }
  sleep 5
done
phase failed
exit 1
