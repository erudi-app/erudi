import { useCallback, useEffect, useRef, useState } from "react";
import { useDownloadModal } from "../../contexts/DownloadModalContext";
import { createLogger } from "../../utils/logger";

const log = createLogger("EngineSwitch");

/**
 * Switching the inference engine: persist the preference, then restart the
 * backend, because the engine is chosen once per boot. Shared by the Settings
 * card and the graphics-card dialog (`EngineFailureModal`).
 *
 * The restart kills a model download in progress, which the next boot marks
 * as interrupted and deletes. So while one runs, `requestSwitch` does not
 * switch: it raises `confirming` and waits. `confirm()` cancels the download
 * through the download widget's own path (no failed poll, the widget reads
 * "cancelled"), then switches; `dismiss()` changes nothing. A download that
 * finished while the question was open is not cancelled. One switch runs at a
 * time: a request made during one joins it, and a confirmation with no
 * question waiting (a second click on a dialog fading out) does nothing.
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
 * `pending` is raised before anything is cancelled or written and drops on
 * every way the switch can end: the preference not saved, no bridge to
 * restart through, the restart refused, or the re-read settled (success or
 * failure).
 *
 * `requestSwitch(next)` resolves to the outcome: `"declined"` (the user kept
 * the download), `"not-saved"`, `"saved"` (no bridge: the choice applies at
 * the next launch), `"restart-failed"` (saved, applies at the next launch) or
 * `"restarted"`.
 */
export function useEngineSwitch({ persist, settled } = {}) {
  const { isDownloading, isCompleting, cancelDownload } = useDownloadModal();
  const [pending, setPending] = useState(false);
  const [confirmingNext, setConfirmingNext] = useState(null);
  const latestSettled = useRef(settled);
  const baseline = useRef(settled);
  const waiting = useRef(false);
  const downloading = useRef(isDownloading);
  const answer = useRef(null);
  const inFlight = useRef(null);

  useEffect(() => {
    downloading.current = isDownloading;
  }, [isDownloading]);

  useEffect(() => {
    latestSettled.current = settled;
    if (waiting.current && settled !== baseline.current) {
      waiting.current = false;
      setPending(false);
    }
  }, [settled]);

  const runSwitch = useCallback(
    async (next, { cancelDownloadFirst = false } = {}) => {
      setPending(true);
      waiting.current = false;
      baseline.current = latestSettled.current;
      // Read at confirmation time, not when the question was asked: a download
      // that has finished since has nothing left to cancel.
      if (cancelDownloadFirst && downloading.current) {
        // Resolves only once the job has ended: a cancel refused because the
        // download is being installed (409) waits for it to complete, so the
        // restart below never lands under its finalization.
        await cancelDownload();
      }
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
    [persist, settled, cancelDownload]
  );

  // One switch at a time: a request made while one runs (a second click
  // before the first re-render) joins it rather than cancelling, saving and
  // restarting a second time.
  const switchTo = useCallback(
    (next, options) => {
      if (inFlight.current) return inFlight.current;
      const run = runSwitch(next, options).finally(() => {
        inFlight.current = null;
      });
      inFlight.current = run;
      return run;
    },
    [runSwitch]
  );

  const requestSwitch = useCallback(
    (next) => {
      if (!downloading.current || inFlight.current) return switchTo(next);
      // A question already open is replaced; its caller hears "declined".
      answer.current?.("declined");
      setConfirmingNext(next);
      return new Promise((resolve) => {
        answer.current = resolve;
      });
    },
    [switchTo]
  );

  const confirm = useCallback(async () => {
    const resolve = answer.current;
    // Already answered: a second click on a dialog that is still fading out
    // calls the confirmation it was rendered with.
    if (!resolve) return undefined;
    const next = confirmingNext;
    answer.current = null;
    setConfirmingNext(null);
    const outcome = await switchTo(next, { cancelDownloadFirst: true });
    resolve?.(outcome);
    return outcome;
  }, [confirmingNext, switchTo]);

  const dismiss = useCallback(() => {
    const resolve = answer.current;
    answer.current = null;
    setConfirmingNext(null);
    resolve?.("declined");
  }, []);

  return {
    pending,
    // The switch waits for a download being installed (its cancel was
    // refused): the UI says so instead of a bare spinner.
    waitingForDownload: Boolean(isCompleting),
    confirming: confirmingNext !== null,
    requestSwitch,
    confirm,
    dismiss,
  };
}
