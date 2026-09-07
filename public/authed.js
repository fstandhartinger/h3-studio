/**
 * Resolve once the browser is actually signed in.
 *
 * The workspace modules load with the page, which is before the password has been typed.
 * Without this they fire their first requests into the login gate, log a row of 401s to
 * the console, and then show "no GPU running" until their next poll happens to land —
 * which reads as the app being wrong about the pod rather than merely early.
 */
export function whenAuthed({ intervalMs = 1000, timeoutMs = 15 * 60 * 1000 } = {}) {
  return new Promise((resolve) => {
    const started = Date.now();
    const check = async () => {
      try {
        const r = await fetch('/api/session', { credentials: 'same-origin' });
        if (r.ok) {
          const j = await r.json();
          if (j.authed || j.noAuth) return resolve();
        }
      } catch { /* offline; try again */ }
      if (Date.now() - started > timeoutMs) return resolve();  // give up waiting, not working
      setTimeout(check, intervalMs);
    };
    check();
  });
}
