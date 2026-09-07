# H3 Studio — front door to a MiniMax H3 / 10Eros-Max ComfyUI pod.
#
# ffmpeg is not optional here: story mode extracts the chaining frame between segments
# and concatenates the finished segments, and both are ffmpeg. The app degrades to
# single-clip mode without it rather than crashing, but the headline feature is gone.
FROM node:22-bookworm-slim

# curl is not optional either, though for an unobvious reason: Coolify runs its own
# healthcheck INSIDE the container with curl or wget and ignores the HEALTHCHECK below,
# so a slim image without one is reported unhealthy and the deploy is rolled back even
# though the app started correctly. That is exactly what happened on the first deploy.
# openssh-client: new pods are provisioned over ssh/scp from inside this container.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl openssh-client \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY server.js ./
COPY lib ./lib
COPY public ./public
COPY provision ./provision

# Cached renders live here. A container restart loses them, which is acceptable:
# the pod they came from is itself hourly, and the gallery is a convenience.
ENV NODE_ENV=production \
    PORT=3000 \
    CACHE_DIR=/app/cache
RUN mkdir -p /app/cache

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:3000/healthz || exit 1

CMD ["node", "server.js"]
