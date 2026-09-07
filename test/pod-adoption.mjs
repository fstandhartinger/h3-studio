/**
 * Regression test for the bug that terminated a pod mid-render.
 *
 * The app adopted a pod it had not rented, could not know its budget, assumed the 1 h
 * default from the pod's creation time, computed a deadline that was already in the past,
 * and stopped the pod one second after startup.
 *
 * The rule under test: a pod that does not state its own deadline is never stopped by
 * this app. Enforcement for those belongs to the Sandy reaper.
 */
import assert from 'node:assert/strict';

process.env.RUNPOD_API_KEY = 'test-key';
process.env.POD_SSH_PRIVATE_KEY = 'x';
process.env.POD_SSH_PUBLIC_KEY = 'y';

const podMod = await import('../lib/pod.js');

let deleted = [];
let pods = [];

globalThis.fetch = async (url, opts = {}) => {
  const u = String(url);
  if (opts.method === 'DELETE') {
    deleted.push(u.split('/').pop());
    return { ok: true, status: 200, text: async () => '{}' };
  }
  return { ok: true, status: 200, text: async () => JSON.stringify(pods) };
};

const HOUR = 3600_000;
const iso = (ms) => new Date(ms).toISOString().replace('T', ' ').replace('Z', ' +0000 UTC');

// ---- 1. a pod rented an hour ago by the CLI, with no H3_DEADLINE ----------
pods = [{
  id: 'oldpod', name: 'minimax-h3-10eros', costPerHr: 2.09,
  createdAt: iso(Date.now() - 1.5 * HOUR),
  env: {},                       // no H3_DEADLINE: the CLI scripts do not set one
}];
deleted = [];
await podMod.refresh();
let st = podMod.status();

assert.equal(st.pod.id, 'oldpod', 'should adopt the running pod');
assert.equal(st.pod.deadline, null, 'an unknown budget must stay unknown, not be guessed');
assert.equal(st.pod.secondsLeft, null, 'no countdown for an unknown budget');
assert.equal(st.pod.unmanaged, true, 'must be flagged as not managed here');
assert.deepEqual(podMod.deadlines(), {}, 'must publish no deadline it invented');

await new Promise((r) => setTimeout(r, 1500));
assert.deepEqual(deleted, [], 'MUST NOT terminate a pod whose budget it does not know');
console.log('✓ an adopted pod with no stated deadline is never auto-stopped');

// ---- 2. extending an unmanaged pod gives it a deadline from NOW ----------
st = podMod.extend(2);
assert.ok(st.pod.deadline > Date.now() + 1.9 * HOUR, 'extend must run from now, not from creation');
assert.equal(st.pod.unmanaged, false, 'it is managed once a deadline is set');
console.log('✓ extending an unmanaged pod sets its deadline from now');

// ---- 3. a pod this app rented, carrying H3_DEADLINE, is honoured ---------
const created = Date.now() - 0.5 * HOUR;
const deadline = created + 1 * HOUR;
pods = [{
  id: 'mypod', name: 'h3-studio-2210', costPerHr: 2.09,
  createdAt: iso(created),
  env: { H3_MANAGED: '1', H3_DEADLINE: String(Math.round(deadline / 1000)) },
}];
deleted = [];
await podMod.refresh();
st = podMod.status();
assert.equal(st.pod.id, 'mypod');
assert.equal(st.pod.unmanaged, false, 'a stated deadline means it is managed');
assert.ok(Math.abs(st.pod.deadline - deadline) < 2000, 'must use the stated deadline');
assert.ok(st.pod.secondsLeft > 1700 && st.pod.secondsLeft < 1810, 'countdown from the stated deadline');
assert.ok(podMod.deadlines().mypod, 'must publish a stated deadline to the reaper');
console.log('✓ a pod stating H3_DEADLINE is honoured and published');

// ---- 4. the cost counter runs from rental, not from readiness -----------
assert.ok(st.pod.costSoFar > 1.0 && st.pod.costSoFar < 1.1,
  `half an hour at $2.09/h should be ~$1.05, got ${st.pod.costSoFar}`);
console.log('✓ cost counter bills from the moment the pod was rented');

console.log('\nall pod-adoption tests passed');
