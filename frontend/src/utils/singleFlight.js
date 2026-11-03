/**
 * Wrap an async function so that only one run is in flight at a time: a call
 * made while a run is pending gets that run's promise instead of starting
 * another. Once the run settles (resolved or rejected) the next call runs
 * again.
 *
 * main.js registers `backend:restart` through it: a restart stops the backend
 * and spawns a new one, and two overlapping restarts would spawn two backends
 * fighting over the embedded PostgreSQL data directory.
 */
export function singleFlight(fn) {
  let inFlight = null;
  return (...args) => {
    if (inFlight) return inFlight;
    // The async wrapper starts `fn` right away and turns a synchronous throw
    // into a rejection like any other.
    inFlight = (async () => fn(...args))().finally(() => {
      inFlight = null;
    });
    return inFlight;
  };
}
