/**
 * Smoke-test the UI against a running server.
 *   node test-ui.mjs [baseUrl] [password]
 * Exercises: load, console errors, status bar, mode tabs, duration snapping,
 * a real generation with SSE progress, and mobile layout. Writes screenshots
 * to ./shots/.
 */
import { chromium } from 'playwright';
import fs from 'node:fs';

const BASE = process.argv[2] || 'http://localhost:3999';
const PASSWORD = process.argv[3] || '';
const SHOTS = new URL('./shots/', import.meta.url).pathname;
fs.mkdirSync(SHOTS, { recursive: true });


// page.screenshot() waits for fonts + stability, which hangs under the
// headless shell in WSL. CDP captureScreenshot has no such wait.
async function shot(p, name, full = false) {
  // Both page.screenshot() and a bare CDP capture can hang under the headless
  // shell in WSL; a screenshot is never worth stalling the run for.
  const cap = (async () => {
    const cdp = await p.context().newCDPSession(p);
    try {
      const { data } = await cdp.send('Page.captureScreenshot', {
        format: 'png', captureBeyondViewport: full,
      });
      fs.writeFileSync(SHOTS + name, Buffer.from(data, 'base64'));
    } finally { await cdp.detach().catch(() => {}); }
  })();
  const timed = new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 12000));
  await Promise.race([cap, timed]).catch((e) =>
    console.log(`  screenshot ${name} skipped (${e.message})`));
}

const problems = [];
const note = (m) => console.log('  ' + m);

const browser = await chromium.launch({ channel: 'chrome' });
// The app honours prefers-reduced-motion; use it so perpetual animations
// (status dot pulse, progress shimmer) don't defeat stability checks.
const ctx = await browser.newContext({
  viewport: { width: 1440, height: 950 },
  reducedMotion: 'reduce',
});
const page = await ctx.newPage();

const consoleErrors = [];
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message));
page.on('requestfailed', (r) => {
  const u = r.url();
  if (!u.startsWith('data:')) consoleErrors.push(`requestfailed: ${u} ${r.failure()?.errorText}`);
});

console.log(`\n== loading ${BASE}`);
await page.goto(BASE, { waitUntil: 'domcontentloaded', timeout: 60000 });

if (PASSWORD) {
  const pw = page.locator('input[type=password]').first();
  if (await pw.count()) {
    note('login screen present, authenticating');
    await pw.fill(PASSWORD);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(2500);
  }
}
await page.waitForTimeout(2500);
await shot(page, '01-desktop.png');

const title = await page.title();
note(`title: ${title}`);

// --- status bar reflects the live pod --------------------------------------
const bodyText = await page.locator('body').innerText();
for (const [label, re] of [
  ['GPU name', /RTX PRO 6000|Blackwell/i],
  ['online state', /online|offline/i],
]) {
  if (!re.test(bodyText)) problems.push(`status bar is missing ${label}`);
  else note(`status bar shows ${label}`);
}

// --- mode tabs --------------------------------------------------------------
for (const label of ['Image', 'Reference', 'Last']) {
  const tab = page.getByRole('tab', { name: new RegExp(label, 'i') })
    .or(page.locator(`button:has-text("${label}")`)).first();
  if (await tab.count()) {
    await tab.click({ timeout: 8000, force: true }).catch(() => problems.push(`tab "${label}" not clickable`));
    await page.waitForTimeout(500);
    note(`tab "${label}" ok`);
  } else problems.push(`no tab matching "${label}"`);
}
await shot(page, '02-reference-mode.png');

// back to text-to-video
const t2v = page.locator('button:has-text("Text")').first();
if (await t2v.count()) { await t2v.click({ force: true }); await page.waitForTimeout(400); }

// --- duration snapping ------------------------------------------------------
const slider = page.locator('input[type=range]').first();
if (await slider.count()) {
  await slider.evaluate((el) => {
    el.value = '5';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.waitForTimeout(400);
  const txt = await page.locator('body').innerText();
  // 5s must snap to 124 frames / 5.17s
  if (/124|5\.1[0-9]/.test(txt)) note('duration snapping to the 17-frame grid is shown');
  else problems.push('duration does not display the snapped frame count');
} else problems.push('no duration slider found');

// --- a real generation ------------------------------------------------------
const ta = page.locator('textarea').first();
if (!(await ta.count())) problems.push('no prompt textarea');
else {
  await ta.fill('A single candle flame flickering in a dark room.\n\nAudio: quiet room tone, a soft crackle.');
  // shrink to the cheapest legal settings so the test is fast
  await page.evaluate(() => {
    const set = (sel, v) => {
      const el = document.querySelector(sel);
      if (!el) return;
      el.value = v;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    };
    for (const el of document.querySelectorAll('input[type=range]')) {
      const max = Number(el.max);
      if (max >= 15 && max <= 16) { el.value = '1'; }        // duration
      else if (max >= 30) { el.value = String(Math.max(Number(el.min) || 4, 6)); } // steps
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }
    set('#width', '640'); set('#height', '352');
  });
  await page.waitForTimeout(500);
  await shot(page, '03-before-generate.png');

  const gen = page.locator('button:has-text("Generate")').first();
  if (!(await gen.count())) problems.push('no Generate button');
  else {
    await gen.click({ force: true });
    note('generation submitted, watching for progress...');
    let sawProgress = false, done = false;
    for (let i = 0; i < 90; i++) {
      await page.waitForTimeout(2000);
      const t = await page.locator('body').innerText();
      if (/\b\d+\s*\/\s*\d+\b|%/.test(t) && /generat|sampl|progress|step/i.test(t)) sawProgress = true;
      const ready = await page.evaluate(() =>
        [...document.querySelectorAll('video')].some((v) => (v.currentSrc || v.src || '').includes('/api/video/')));
      if (ready) { done = true; break; }
      if (/error|failed/i.test(t) && !/no error/i.test(t)) {
        const snippet = t.split('\n').filter((l) => /error|failed/i.test(l)).slice(0, 3).join(' | ');
        problems.push(`UI reported an error: ${snippet}`);
        break;
      }
    }
    if (!sawProgress) problems.push('never showed sampling progress');
    else note('progress was displayed');
    if (!done) problems.push('no <video> element appeared within 180s');
    else {
      note('video element rendered');
      // chrome-headless-shell ships without an H.264 decoder, so videoWidth is
      // always 0 here. Check that the src resolves and the server serves a real
      // mp4 instead of trying to decode it.
      const src = await page.evaluate(() => {
        const v = [...document.querySelectorAll('video')]
          .find((x) => (x.currentSrc || x.src || '').includes('/api/video/'));
        return v ? (v.currentSrc || v.src) : '';
      });
      if (!src) problems.push('video element has no src');
      else {
        const r = await page.request.get(new URL(src, BASE).href);
        const ct = r.headers()['content-type'] || '';
        const len = Number(r.headers()['content-length'] || 0);
        note(`video src ${src} -> ${r.status()} ${ct} ${len}B`);
        if (r.status() !== 200) problems.push(`video url returned ${r.status()}`);
        if (!/mp4/.test(ct)) problems.push(`video content-type is "${ct}"`);
        if (len < 5000) problems.push(`video is only ${len} bytes`);
      }
      const hasAudioHint = /unmute|audio|sound|🔊/i.test(await page.locator('body').innerText());
      if (!hasAudioHint) problems.push('no hint that the model generated the audio');
      else note('audio affordance present');
    }
    await shot(page, '04-result.png');
  }
}

// --- mobile -----------------------------------------------------------------
const mob = await ctx.newPage();
await mob.setViewportSize({ width: 390, height: 844 });
await mob.goto(BASE, { waitUntil: 'domcontentloaded' });
await mob.waitForTimeout(2000);
const overflow = await mob.evaluate(() =>
  document.documentElement.scrollWidth - document.documentElement.clientWidth);
if (overflow > 4) problems.push(`mobile has ${overflow}px of horizontal overflow`);
else note('mobile: no horizontal overflow');
await shot(mob, '05-mobile.png');

// --- light theme ------------------------------------------------------------
await page.evaluate(() => {
  const b = [...document.querySelectorAll('button')].find((x) =>
    /theme|light|dark/i.test(x.getAttribute('aria-label') || x.title || x.textContent || ''));
  b?.click();
});
await page.waitForTimeout(700);
await shot(page, '06-light.png');

if (consoleErrors.length) {
  problems.push(`console errors: ${[...new Set(consoleErrors)].slice(0, 6).join(' || ')}`);
}

console.log('\n== result');
if (problems.length === 0) console.log('  ALL CHECKS PASSED');
else problems.forEach((p) => console.log('  PROBLEM: ' + p));
console.log(`  screenshots in ${SHOTS}`);

await browser.close();
process.exit(problems.length ? 1 : 0);
