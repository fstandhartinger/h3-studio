/**
 * Human names for the provisioning phases, shared by the pod bar and the offline banner.
 *
 * The raw phase words ("comfyui", "weights") are what the installer writes to
 * /workspace/phase; they are a protocol, not a UI. Nobody watching a 15-minute install
 * should have to know that "comfyui" means pip is running while the models download.
 */
export const PHASES = [
  { key: 'booting',            label: 'Renting the GPU' },
  { key: 'waiting for ssh',    label: 'Waiting for the machine to boot' },
  { key: 'uploading installer',label: 'Uploading the installer' },
  { key: 'installing',         label: 'Starting the installer' },
  { key: 'apt',                label: 'Installing system packages' },
  { key: 'comfyui',            label: 'Installing ComfyUI while the models download' },
  { key: 'weights',            label: 'Downloading models (66 GB)' },
  { key: 'verify',             label: 'Verifying every checkpoint' },
  { key: 'starting',           label: 'Starting ComfyUI' },
  { key: 'ready',              label: 'Ready' },
];

/** Typical wall clock, measured: 12 min on a good day, 16 on a slow PyPI day. */
export const TYPICAL_MINUTES = '12–16';

export function describePhase(raw) {
  const i = PHASES.findIndex((p) => p.key === raw);
  if (i === -1) return { label: raw || 'starting…', step: 0, total: PHASES.length - 1, pct: 0 };
  const total = PHASES.length - 1;                 // "ready" is the finish line, not a step
  const step = Math.min(i + 1, total);
  return { label: PHASES[i].label, step, total, pct: Math.round((i / total) * 100) };
}

export function fmtMinutes(sec) {
  const m = Math.max(0, Math.round(sec / 60));
  return m === 1 ? '1 min' : `${m} min`;
}
