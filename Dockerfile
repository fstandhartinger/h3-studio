# H3 Studio — front door to a MiniMax H3 / 10Eros-Max ComfyUI pod.
#
# ffmpeg is not optional here: story mode extracts the chaining frame between segments
# and concatenates the finished segments, and both are ffmpeg. The app degrades to
# single-clip mode without it rather than crashing, but the headline feature is gone.
FROM node:22-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY server.js ./
COPY lib ./lib
COPY public ./public

# Cached renders live here. A container restart loses them, which is acceptable:
# the pod they came from is itself hourly, and the gallery is a convenience.
ENV NODE_ENV=production \
    PORT=3000 \
    CACHE_DIR=/app/cache
RUN mkdir -p /app/cache

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3000/healthz').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
