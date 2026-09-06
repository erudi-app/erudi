import { useEffect, useMemo, useState } from "react";
import { apiClient } from "../../services/api/client";
import { mergeRecentErrors } from "../../utils/diagnosticsReport";
import {
  countNewErrors,
  getLastDiagnosticsVisit,
  subscribeDiagnosticsVisit,
  badgeLabel,
} from "../../utils/bugCounter";
import { getSessionErrors, subscribeSessionErrors } from "../../utils/errorCapture";

/**
 * How many new, real error records the bug icon in the sidebar should show
 * (#485).
 *
 * Reuses the exact three sources and the exact merge the Diagnostics page
 * reads (`mergeRecentErrors`, `utils/diagnosticsReport.js`) so the count
 * always matches what opening the page would list, then narrows it with
 * `utils/bugCounter.js`'s predicate to ERROR/CRITICAL records that are not
 * environmental and are newer than the later of app launch and the last
 * Diagnostics visit.
 *
 * Two update paths, deliberately different in cost:
 *   - The backend log and the app log only change between polls of >= 60s
 *     (`pollMs`) -- one `apiClient.get("/diagnostics/")` (same call
 *     DiagnosticsPanel makes, so an unreachable backend degrades exactly the
 *     way that page already does) and one `diagnosticsAPI.appLogTail` call,
 *     which answers `{ok, records}` -- a log it could not read leaves the
 *     count as it was instead of clearing the badge.
 *     This is the "gentle poll" the feature asks for; ConnectionStatus
 *     already polls health every 15s for a different purpose, so this stays
 *     independent and much slower.
 *   - Session errors (already in memory, `utils/errorCapture.js`) and a
 *     Diagnostics visit (`utils/bugCounter.js`) both update the badge the
 *     instant they happen, through their own subscriptions -- no reason to
 *     make a render loop or "I just opened the page" wait up to a minute.
 *
 * @param {object} [options]
 * @param {number} [options.pollMs] - Poll interval; overridable for tests only.
 * @returns {{count: number, label: string}} `label` is `badgeLabel(count)`
 *   ("", "3", "9+").
 */

/** Minimum poll interval this hook will use: a gentle default. */
export const BUG_COUNTER_POLL_MS = 60000;

const APP_LOG_TAIL_LIMIT = 200;

// Captured once, when the renderer first loads this module during boot -- a
// process-local approximation of "app launch" good enough for this feature.
// A value read fresh on every call would defeat the point: nothing would
// ever be older than "now" and every record would count as new forever.
const APP_LAUNCH_TIME = Date.now();

export default function useBugCounter({ pollMs = BUG_COUNTER_POLL_MS } = {}) {
  const [backend, setBackend] = useState(null);
  const [appLog, setAppLog] = useState([]);
  const [sessionErrors, setSessionErrorsState] = useState(() => getSessionErrors());
  const [lastVisit, setLastVisit] = useState(() => getLastDiagnosticsVisit());

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      if (cancelled) return;

      // Every source settles independently, same as DiagnosticsPanel: the
      // backend being unreachable must not blank out the app-log half, and
      // vice versa.
      //
      // `silentFailure` matters here specifically: this poll is the counter
      // OBSERVING /diagnostics/, not a person asking for it. Without it, a
      // down backend would log this very failure as an ERROR api.failure,
      // which the next tick's `appLogTail` read would feed straight back
      // into `entries` below -- the counter creating the signal it counts,
      // growing every interval forever. A real user-initiated call to
      // /diagnostics/ (there is none today, but if one existed) would not
      // set this flag and would still count once, like any other failure.
      const [backendResult, logResult] = await Promise.allSettled([
        apiClient.get("/diagnostics/", { silentFailure: true }),
        window.diagnosticsAPI?.appLogTail?.(APP_LOG_TAIL_LIMIT) ??
          Promise.resolve({ ok: true, records: [] }),
      ]);

      if (cancelled) return;
      if (backendResult.status === "fulfilled") setBackend(backendResult.value);
      // `{ok, records}`: a log that could not be read leaves the previous
      // count standing rather than dropping the badge to zero, which would
      // read as "the errors went away". The panel is where the failure is
      // reported; a silent badge must not invent good news.
      const logValue = logResult.status === "fulfilled" ? logResult.value : null;
      if (logValue?.ok && Array.isArray(logValue.records)) {
        setAppLog(logValue.records);
      }
    }

    poll();
    const interval = setInterval(poll, pollMs);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [pollMs]);

  useEffect(() => subscribeSessionErrors(() => setSessionErrorsState(getSessionErrors())), []);

  useEffect(() => subscribeDiagnosticsVisit((now) => setLastVisit(now)), []);

  const entries = useMemo(
    () => mergeRecentErrors({ backend, appLog, sessionErrors }),
    [backend, appLog, sessionErrors]
  );

  const since = Math.max(APP_LAUNCH_TIME, lastVisit);
  const count = useMemo(() => countNewErrors({ entries, since }), [entries, since]);

  return { count, label: badgeLabel(count) };
}
