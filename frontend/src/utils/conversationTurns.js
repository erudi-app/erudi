// The answers streaming now, per conversation id, and their AbortControllers.
// The conversation page registers each answer it streams (`trackTurn`); a
// delete aborts them first (`abortTurns`), from the history row or when the
// user leaves an ephemeral conversation (#428). The backend waits for a turn
// still writing to a conversation before it deletes it, and the abort is what
// ends that turn at once instead of at the end of the generation.

// Ids are kept as strings: the route's form.
const turnsInFlight = new Map();

const key = (id) => String(id);

/**
 * Register an answer streaming for conversation `id`.
 * @returns {Function} Unregisters it (call when the answer is over).
 */
export function trackTurn(id, controller) {
  const k = key(id);
  if (!turnsInFlight.has(k)) turnsInFlight.set(k, new Set());
  turnsInFlight.get(k).add(controller);
  return () => {
    const controllers = turnsInFlight.get(k);
    if (!controllers) return;
    controllers.delete(controller);
    if (controllers.size === 0) turnsInFlight.delete(k);
  };
}

/** Abort every answer streaming for conversation `id` (idempotent). */
export function abortTurns(id) {
  const k = key(id);
  for (const controller of turnsInFlight.get(k) ?? []) controller.abort();
  turnsInFlight.delete(k);
}

/** Test helper: forget every answer. */
export function resetConversationTurns() {
  turnsInFlight.clear();
}
