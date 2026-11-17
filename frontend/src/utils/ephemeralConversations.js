import { matchPath } from "react-router-dom";
import { API_BASE_URL } from "../config/api.js";
import { tracedFetch } from "../services/api/client";
import { abortTurns, resetConversationTurns } from "./conversationTurns";
import { createLogger } from "./logger";

// Ephemeral conversations (#428): the renderer's half. A conversation created
// with the ephemeral toggle is deleted when the user leaves it -- when the
// conversation id in the route changes (another conversation, the Chat page,
// any other page), and when the window closes or reloads (`pagehide`).
// `EphemeralConversationGuard` (mounted once, inside the router) watches both
// and calls `leaveConversation`; the conversation page marks the
// conversations it shows as ephemeral. Leaving aborts the conversation's
// streaming answer (utils/conversationTurns.js) before the DELETE goes out.
// The backend waits for that turn to end before it deletes anything, and
// deletes what the renderer could not (the app quit, a crash) at shutdown
// and at the next start.

const log = createLogger("ephemeralConversations");

const CONVERSATION_ROUTE = "/erudi/conversations/:id";

// Ids are kept as strings: the route's form.
const ephemeralIds = new Set();

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
 * The user left conversation `id`. An ephemeral one has its streaming answer
 * aborted and is deleted (no confirmation); any other is left as it is.
 * Idempotent: a conversation is purged once.
 *
 * `keepalive` is for a window closing or reloading (`pagehide`): the request
 * outlives the page, and nothing waits for its answer.
 * @returns {Promise<boolean>} Whether a delete was sent.
 */
export async function leaveConversation(id, { fetcher = tracedFetch, keepalive = false } = {}) {
  const k = key(id);
  if (!ephemeralIds.delete(k)) return false;
  abortTurns(k);
  const url = `${API_BASE_URL}/conversations/${k}`;
  if (keepalive) {
    // The page is going away: the backend's sweeps delete it if this request
    // never arrives.
    fetcher(url, { method: "DELETE", keepalive: true }).catch((error) => {
      log.warn(`Ephemeral conversation ${k} not deleted as the window closed`, error);
    });
    return true;
  }
  try {
    const response = await fetcher(url, { method: "DELETE" });
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
  resetConversationTurns();
}
