import { matchPath } from "react-router-dom";
import { API_BASE_URL } from "../config/api.js";
import { tracedFetch } from "../services/api/client";
import { createLogger } from "./logger";

// Ephemeral conversations (#428): the renderer's half. A conversation created
// with the ephemeral toggle is deleted when the user leaves it -- when the
// conversation id in the route changes (another conversation, the Chat page,
// any other page). `EphemeralConversationGuard` (mounted once, inside the
// router) watches the route and calls `leaveConversation`; the conversation
// page marks the conversations it shows as ephemeral and registers the answer
// it streams, so leaving aborts it before the DELETE goes out. The backend
// waits for that turn to end before it deletes anything, and deletes what the
// renderer could not (the app quit, a crash) at shutdown and at the next start.

const log = createLogger("ephemeralConversations");

const CONVERSATION_ROUTE = "/erudi/conversations/:id";

// Ids are kept as strings: the route's form.
const ephemeralIds = new Set();
// The answers streaming now, per conversation id: their AbortControllers.
const turnsInFlight = new Map();

const key = (id) => String(id);

/** The conversation id of a route, or null when the route shows none. */
export function conversationIdFromPath(pathname) {
  return matchPath(CONVERSATION_ROUTE, pathname || "")?.params.id ?? null;
}

/** Remember that this conversation is ephemeral (idempotent). */
export function markEphemeral(id) {
  ephemeralIds.add(key(id));
}

export function isEphemeral(id) {
  return ephemeralIds.has(key(id));
}

/**
 * Register the answer streaming for `id`; leaving the conversation aborts it.
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

/**
 * The user left conversation `id`. An ephemeral one has its streaming answer
 * aborted and is deleted (no confirmation); any other is left as it is.
 * Idempotent: a conversation is purged once.
 * @returns {Promise<boolean>} Whether a delete was sent.
 */
export async function leaveConversation(id, { fetcher = tracedFetch } = {}) {
  const k = key(id);
  if (!ephemeralIds.delete(k)) return false;
  for (const controller of turnsInFlight.get(k) ?? []) controller.abort();
  turnsInFlight.delete(k);
  try {
    const response = await fetcher(`${API_BASE_URL}/conversations/${k}`, { method: "DELETE" });
    if (response.ok || response.status === 404) {
      log.info(`Ephemeral conversation ${k} deleted on leave`);
    } else {
      // The backend deletes it at shutdown or at the next start; it is never
      // listed meanwhile.
      log.warn(`Ephemeral conversation ${k} not deleted on leave (HTTP ${response.status})`);
    }
  } catch (error) {
    log.warn(`Ephemeral conversation ${k} not deleted on leave; the backend sweeps it`, error);
  }
  return true;
}

/** Test helper: forget every conversation and answer. */
export function resetEphemeralConversations() {
  ephemeralIds.clear();
  turnsInFlight.clear();
}
