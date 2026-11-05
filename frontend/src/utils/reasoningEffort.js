/**
 * The five reasoning-effort levels a conversation (or the global default)
 * can carry (backend `src/agents/reasoning_effort.py:REASONING_EFFORT_LEVELS`,
 * mirrored here so the frontend has one source of truth, in order: the
 * notches of the /effort and /config slider and the Settings page options).
 */
export const REASONING_EFFORT_LEVELS = ["none", "low", "medium", "high", "xhigh"];

export const DEFAULT_REASONING_EFFORT = "medium";
