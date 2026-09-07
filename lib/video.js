/**
 * ffmpeg helpers for story mode: pulling a chaining frame out of a finished segment,
 * and stitching the segments into one film.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fsp from 'node:fs/promises';
import path from 'node:path';

const execFileP = promisify(execFile);

const FFMPEG = process.env.FFMPEG_PATH || 'ffmpeg';
const FFPROBE = process.env.FFPROBE_PATH || 'ffprobe';

export async function ffmpegAvailable() {
  try {
    await execFileP(FFMPEG, ['-version'], { timeout: 10000 });
    return true;
  } catch {
    return false;
  }
}

export async function probeDuration(file) {
  const { stdout } = await execFileP(FFPROBE, [
    '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', file,
  ], { timeout: 30000 });
  const d = Number(String(stdout).trim());
  return Number.isFinite(d) ? d : null;
}

/**
 * Extract the frame that the NEXT segment should start from.
 *
 * This is the whole reason story mode looks continuous, and it is where this pipeline
 * differs from the earlier fal experiment. That one generated every keyframe with an
 * image model and asked it to keep the characters consistent; identity drifted a
 * little on every hop, because each keyframe was a fresh guess at the same face.
 *
 * Here the next segment starts from pixels the video model itself produced, so the
 * face, wardrobe, lighting and set do not drift at all — they are literally the same
 * frame.
 *
 * `offsetSec` backs off slightly from the true final frame on purpose: the last frame
 * of a diffusion-sampled clip is often the most motion-blurred one, and handing a
 * blurred frame to the next segment bakes that blur into its opening. ~0.12 s (about
 * three frames at 24 fps) is enough to land on a clean frame without visibly skipping.
 */
export async function extractChainFrame(videoFile, outFile, { offsetSec = 0.12 } = {}) {
  const duration = await probeDuration(videoFile);
  const args = [];
  if (duration && duration > offsetSec) {
    // Seek from the end. Accurate enough here and much faster than decoding the file.
    args.push('-sseof', String(-Math.max(offsetSec, 1 / 24)));
  }
  args.push('-i', videoFile, '-frames:v', '1', '-q:v', '2', '-y', outFile);
  await execFileP(FFMPEG, args, { timeout: 120000 });
  const st = await fsp.stat(outFile);
  if (!st.size) throw new Error('extracted chain frame is empty');
  return outFile;
}

/**
 * Concatenate segments into one mp4, video and audio together.
 *
 * Stream copy is tried first: every segment comes out of the same ComfyUI graph at the
 * same resolution and fps, so the encoder parameters normally match and copying is
 * both instant and lossless. It still falls back to a re-encode, because a mismatch
 * that copy silently turns into a broken file is worse than spending 30 s on x264 —
 * and H3's audio track length is not always exactly the video length, which is the
 * usual thing that trips the demuxer.
 */
export async function concatVideos(files, outFile, { workDir } = {}) {
  if (!files.length) throw new Error('nothing to concatenate');
  if (files.length === 1) {
    await fsp.copyFile(files[0], outFile);
    return { outFile, method: 'copy-single' };
  }

  const dir = workDir || path.dirname(outFile);
  const listFile = path.join(dir, `concat-${Date.now()}.txt`);
  // Paths are quoted for the demuxer's own parser; single quotes must be escaped.
  await fsp.writeFile(listFile, files
    .map((f) => `file '${path.resolve(f).replace(/'/g, "'\\''")}'`)
    .join('\n'));

  const expected = (await Promise.all(files.map(probeDuration)))
    .reduce((a, b) => a + (b || 0), 0);

  try {
    await execFileP(FFMPEG, [
      '-f', 'concat', '-safe', '0', '-i', listFile,
      '-c', 'copy', '-movflags', '+faststart', '-y', outFile,
    ], { timeout: 300000 });
    const got = await probeDuration(outFile);
    // A demuxer concat that drops a segment still exits 0, so check the duration.
    if (got && expected && Math.abs(got - expected) <= Math.max(0.75, expected * 0.05)) {
      await fsp.unlink(listFile).catch(() => {});
      return { outFile, method: 'stream-copy', durationSec: got };
    }
  } catch { /* fall through to the re-encode */ }

  const inputs = files.flatMap((f) => ['-i', f]);
  const n = files.length;
  const filter = files.map((_, i) => `[${i}:v:0][${i}:a:0]`).join('')
    + `concat=n=${n}:v=1:a=1[v][a]`;
  await execFileP(FFMPEG, [
    ...inputs,
    '-filter_complex', filter, '-map', '[v]', '-map', '[a]',
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-b:a', '192k',
    '-movflags', '+faststart', '-y', outFile,
  ], { timeout: 1800000, maxBuffer: 16 * 1024 * 1024 });

  await fsp.unlink(listFile).catch(() => {});
  return { outFile, method: 're-encode', durationSec: await probeDuration(outFile) };
}
