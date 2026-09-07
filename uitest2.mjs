import { chromium } from 'playwright';
const PW = 'e34vp-nc93a-uda25-dq45r';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1560, height: 1100 } });
const errs = [];
p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
p.on('pageerror', e => errs.push('PAGEERROR: ' + e.message));

await p.goto('https://h3-studio.app.mintapis.com', { waitUntil: 'networkidle' });
await p.fill('#login-password', PW);
await p.click('#login-submit');
await p.waitForSelector('#app:not([hidden])', { timeout: 20000 });
await p.waitForTimeout(4000);

// Pod bar must show the live pod, not the idle form.
const idleHidden = await p.isHidden('#pod-idle');
const timer = await p.textContent('#pod-timer');
const cost = await p.textContent('#pod-cost');
const phase = await p.textContent('#pod-phase');
console.log(`podbar: idle hidden=${idleHidden} phase="${phase}" timer=${timer} cost=${cost}`);

// the timer must actually tick
const t1 = await p.textContent('#pod-timer');
await p.waitForTimeout(3000);
const t2 = await p.textContent('#pod-timer');
console.log(`timer ticks: ${t1} -> ${t2}  ${t1 !== t2 ? 'OK' : 'NOT TICKING'}`);

await p.screenshot({ path: 'ui-podbar.png', clip: { x: 0, y: 0, width: 1560, height: 200 } });

// Storyboard tab
await p.click('#vtab-board');
await p.waitForTimeout(1200);
console.log('board visible =', await p.isVisible('#board-premise'),
            '| clip hidden =', await p.isHidden('#view-clip'),
            '| story hidden =', await p.isHidden('#view-story'));
console.log('chroma warning shown =', await p.isVisible('#board-nochroma'), '(should be false)');
console.log('maths line =', await p.textContent('#board-maths'));
await p.fill('#board-keyframes', '6');
await p.dispatchEvent('#board-keyframes', 'input');
await p.waitForTimeout(300);
console.log('maths after 6 keyframes =', await p.textContent('#board-maths'));
await p.screenshot({ path: 'ui-board.png' });

// Story tab still works
await p.click('#vtab-story');
await p.waitForTimeout(800);
console.log('story visible =', await p.isVisible('#story-premise'), '| board hidden =', await p.isHidden('#view-board'));

console.log('console errors:', errs.length ? errs : 'none');
await b.close();
