import React, { useState, useEffect, useCallback, useRef } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import { HelpCircle, RefreshCw } from "lucide-react";
import Tooltip from "./Tooltip";
import { getApiBaseUrl } from "../config/api";
import { isNetworkOnline, subscribeNetworkStatus } from "../utils/networkStatus";
import { createLogger } from "../utils/logger";

const log = createLogger("ConnectionStatus");

/**
 * Live status pill for the bottom of the left rail.
 *
 * Two signals, neither of which costs a request to anyone else:
 *
 *   - GET /health/ -> {status, db: "ok"|"recovering"|"failed"}
 *       Local and cheap; polled every ~15s. Drives the backend-reachable check
 *       and surfaces the DB watchdog state added in #162/#270.
 *   - Connectivity, from `navigator.onLine` plus the `online` / `offline`
 *       events, corrected by the requests the app already makes (see
 *       utils/networkStatus). The OS answers instantly and for free; a request
 *       that dies at the network layer overrides it, because a link is not the
 *       same thing as a reachable internet.
 *
 * Display priority (offline is NOT an error -- local chat keeps working):
 *   1. backend unreachable (health poll fails/times out) -> red + Restart
 *   2. db === "recovering"                                -> amber "Restoring the database..."
 *   3. db === "failed"                                    -> red "Database error" + Restart
 *   4. no network                                         -> gray "Offline" (informative)
 *   5. all good                                           -> green "Connected"
 */

// Poll cadence and per-request client timeout for the local health check.
// Exposed as props so tests can drive the state machine on short intervals;
// production uses the defaults.
const HEALTH_POLL_MS = 15000;
const HEALTH_TIMEOUT_MS = 8000;

/**
 * Map the raw signals to a single visual descriptor, honoring the priority
 * order. `state` names the `errors:connection.<state>` subtree (label, title,
 * tooltip) the component translates at render time.
 */
function resolveDisplay({ backendReachable, dbState, online }) {
  if (!backendReachable) {
    return {
      state: "unreachable",
      dot: "bg-red-500",
      labelClass: "text-red-300",
      showRestart: true,
    };
  }
  if (dbState === "recovering") {
    return {
      state: "recovering",
      dot: "bg-amber-400",
      ping: "bg-amber-400/60",
      pulse: true,
      labelClass: "text-amber-300",
    };
  }
  if (dbState === "failed") {
    return {
      state: "databaseError",
      dot: "bg-red-500",
      labelClass: "text-red-300",
      showRestart: true,
    };
  }
  if (online === false) {
    return {
      state: "offline",
      dot: "bg-gray-500",
      labelClass: "text-gray-400",
    };
  }
  return {
    state: "connected",
    dot: "bg-emerald-400",
    ping: "bg-emerald-400/60",
    pulse: true,
    labelClass: "text-gray-300",
  };
}

export default function ConnectionStatus({
  healthPollMs = HEALTH_POLL_MS,
  healthTimeoutMs = HEALTH_TIMEOUT_MS,
}) {
  const { t } = useTranslation();
  // Optimistic backend defaults so mounting never flashes an alarming state
  // before the first health poll resolves. Connectivity needs no such guess:
  // the operating system already knows the answer.
  const [status, setStatus] = useState({
    backendReachable: true,
    dbState: "ok",
    online: isNetworkOnline(),
  });

  // Kept in a ref so the interval callback always sees the latest timeout
  // without re-subscribing the effect on every prop identity change.
  const healthTimeoutRef = useRef(healthTimeoutMs);
  healthTimeoutRef.current = healthTimeoutMs;

  useEffect(() => {
    let cancelled = false;
    const controllers = new Set();
    // Guards against overlapping requests (a slow poll must not stack on top of
    // the next interval tick).
    let inFlight = false;

    async function pollHealth() {
      if (cancelled || inFlight) return;
      inFlight = true;
      const controller = new AbortController();
      controllers.add(controller);
      const timer = setTimeout(() => controller.abort(), healthTimeoutRef.current);
      try {
        const res = await fetch(`${getApiBaseUrl()}/health/`, {
          signal: controller.signal,
          cache: "no-store",
        });
        if (!res.ok) throw new Error(`health ${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        const db = data && typeof data.db === "string" ? data.db : "ok";
        setStatus((s) => ({ ...s, backendReachable: true, dbState: db }));
      } catch {
        // Any failure/timeout means the local backend is not answering. A failed
        // poll is a state change, never a crash.
        if (!cancelled) setStatus((s) => ({ ...s, backendReachable: false }));
      } finally {
        clearTimeout(timer);
        controllers.delete(controller);
        inFlight = false;
      }
    }

    pollHealth();
    const healthTimer = setInterval(pollHealth, healthPollMs);

    return () => {
      cancelled = true;
      clearInterval(healthTimer);
      controllers.forEach((c) => c.abort());
    };
  }, [healthPollMs]);

  // Connectivity: no request of our own, ever. The OS reports link changes as
  // they happen and the API client reports requests that died on the wire.
  useEffect(() => {
    setStatus((s) => ({ ...s, online: isNetworkOnline() }));
    return subscribeNetworkStatus((online) => setStatus((s) => ({ ...s, online })));
  }, []);

  const handleRestart = useCallback(() => {
    const pending = window.backendAPI?.restartBackend?.();
    if (pending && typeof pending.catch === "function") {
      pending.catch((error) => {
        log.error("The backend restart the user asked for did not happen", error);
      });
    }
  }, []);

  const d = resolveDisplay(status);

  return (
    <div
      className="flex items-center gap-2.5 px-4 py-3 border-t border-white/10"
      title={t(`errors:connection.${d.state}.title`)}
    >
      <span className="relative flex w-2.5 h-2.5">
        {d.pulse && (
          <span
            className={`absolute inline-flex w-full h-full rounded-full ${d.ping} animate-ping`}
          />
        )}
        <span className={`relative inline-flex w-2.5 h-2.5 rounded-full ${d.dot}`} />
      </span>
      <span className={`text-sm ${d.labelClass}`}>{t(`errors:connection.${d.state}.label`)}</span>
      {d.showRestart && (
        <button
          type="button"
          onClick={handleRestart}
          title={t("errors:connection.restartTitle")}
          className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-gray-300 hover:text-white hover:bg-white/10 transition-colors"
        >
          <RefreshCw className="w-3 h-3" />
          {t("errors:connection.restart")}
        </button>
      )}
      <Tooltip side="top-right" width="w-64" content={t(`errors:connection.${d.state}.tooltip`)}>
        <HelpCircle className="w-3.5 h-3.5 text-gray-400 hover:text-emerald-400 transition-colors cursor-help" />
      </Tooltip>
    </div>
  );
}

ConnectionStatus.propTypes = {
  healthPollMs: PropTypes.number,
  healthTimeoutMs: PropTypes.number,
};
