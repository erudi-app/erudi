import { useCallback, useEffect, useRef, useState } from "react";
import { createLogger } from "../../utils/logger";

const log = createLogger("EngineSwitch");

/**
 * Switching the inference engine: persist the preference, then restart the
 * backend, because the engine is chosen once per boot.
 *
 * `persist(next)` writes `inference_backend` and rejects when it did not; the
 * caller owns the record of that failure (it knows which request it made).
 * `settled` is a value that changes once the app has re-read the running
 * engine after the restart (`useAppStartupInfo().refetchesAfterReady`); when
 * it is given, `pending` holds until it changes, so a screen that depends on
 * the running engine keeps its state through the restart. Without it, or
 * without the Electron bridge's event channel, `pending` ends with the
 * restart call.
 *
 * `pending` is raised before the preference is written and drops on every way
 * the switch can end: the preference not saved, no bridge to restart through,
 * the restart refused, or the re-read settled (success or failure).
 *
 * `switchTo(next)` resolves to the outcome: `"not-saved"`, `"saved"` (no
 * bridge: the choice applies at the next launch), `"restart-failed"` (saved,
 * applies at the next launch) or `"restarted"`.
 */
export function useEngineSwitch({ persist, settled } = {}) {
  const [pending, setPending] = useState(false);
  const latestSettled = useRef(settled);
  const baseline = useRef(settled);
  const waiting = useRef(false);

  useEffect(() => {
    latestSettled.current = settled;
    if (waiting.current && settled !== baseline.current) {
      waiting.current = false;
      setPending(false);
    }
  }, [settled]);

  const switchTo = useCallback(
    async (next) => {
      setPending(true);
      waiting.current = false;
      baseline.current = latestSettled.current;
      try {
        await persist(next);
      } catch {
        // Recorded by `persist`'s owner. Restarting for a preference that was
        // not saved would only lose the user's conversation state.
        setPending(false);
        return "not-saved";
      }
      const bridge = window.backendAPI;
      if (typeof bridge?.restartBackend !== "function") {
        // Outside Electron there is nothing to restart: the choice is saved
        // and applies at the next launch.
        setPending(false);
        return "saved";
      }
      try {
        await bridge.restartBackend();
      } catch (error) {
        // A distinct failure with a distinct consequence: the choice IS saved,
        // and it applies at the next launch. Reporting it as a failed setting
        // sent people looking for a defect in the wrong place.
        log.error("The inference engine was saved but the backend did not restart", error);
        setPending(false);
        return "restart-failed";
      }
      // The restart call resolves once the old backend is gone, before the new
      // one is ready (main.js does not await the respawn): wait for the re-read
      // when there is one to wait for.
      if (settled === undefined || typeof bridge.onBackendEvent !== "function") {
        setPending(false);
      } else if (latestSettled.current !== baseline.current) {
        setPending(false);
      } else {
        waiting.current = true;
      }
      return "restarted";
    },
    [persist, settled]
  );

  return { pending, switchTo };
}
