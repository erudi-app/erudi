// The conversations a /compact is running for (#585), kept for the whole
// renderer session: leaving a conversation's page and coming back while its
// summary runs still shows it locked, with its status row. A reload of the
// window forgets them (the backend's own guard still refuses a second run).

const running = new Set();
const listeners = new Set();
let version = 0;

const notify = (conversationId) => {
  version += 1;
  listeners.forEach((listener) => listener(conversationId));
};

/** Whether a /compact runs for `conversationId`. */
export function isCompacting(conversationId) {
  return running.has(String(conversationId));
}

/**
 * Marks a /compact as running for `conversationId`. False when one already
 * runs for it: the caller starts nothing.
 */
export function startCompacting(conversationId) {
  const key = String(conversationId);
  if (running.has(key)) return false;
  running.add(key);
  notify(key);
  return true;
}

/** Marks the /compact of `conversationId` as ended. */
export function finishCompacting(conversationId) {
  const key = String(conversationId);
  if (running.delete(key)) notify(key);
}

/**
 * Calls `listener(conversationId)` whenever a run starts or ends; returns the
 * unsubscribe function (useSyncExternalStore's contract).
 */
export function subscribeCompacting(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/**
 * Forgets every run, telling the listeners (for tests: a test that leaves a
 * run unfinished must not lock its conversation for the next one).
 */
export function resetCompacting() {
  [...running].forEach(finishCompacting);
}

/** Changes whenever a run starts or ends (useSyncExternalStore's snapshot). */
export function compactingVersion() {
  return version;
}
