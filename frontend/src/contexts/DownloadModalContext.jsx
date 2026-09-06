// src/contexts/DownloadModalContext.jsx
import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from "react";
import ReactDOM from "react-dom";
import { AnimatePresence } from "framer-motion";
import { useTranslation } from "react-i18next";
import ConfirmationModal from "../components/modals/ConfirmationModal";
import DownloadWidget from "../components/DownloadWidget";
import { API_BASE_URL } from "../config/api.js";
import { tracedFetch } from "../services/api/client";
import { createLogger } from "../utils/logger";
import {
  DOWNLOAD_CANCELLED,
  DOWNLOAD_STALLED,
  deriveDownloadPhase,
  downloadFailureDetail,
  downloadStalledMessage,
} from "../utils/downloadStatus";
const log = createLogger("DownloadModalContext");

// Poll cadence, and the stall guard on top of it (#315).
//
// The poll used to stop ONLY on a `completed` / `failed` / `cancelled` reply, so
// the day the backend stopped producing one (#291 strands the job at `running`
// with progress 100) it ran forever: 800+ requests over 25 minutes with the
// widget frozen, the sidebar dimmed and no way out of the app.
//
// The cap is scoped to FINALIZATION, not to the download as a whole. A transfer
// that is merely slow is still a healthy transfer -- a 70 GB pull on a bad line
// legitimately takes hours -- so it must never be killed by a timer. Once
// progress reaches 100% the bytes are in, and everything after that is local
// bookkeeping that should take seconds; if that has not moved for
// FINALIZE_STALL_TICKS consecutive polls, the job is not coming back on its own.
const POLL_INTERVAL_MS = 2000;
const FINALIZE_STALL_TICKS = 90; // 90 x 2s = 3 minutes stuck at 100%

// The widget starts as a pill and opens by itself once the confirmation has
// gone and the first poll is about to land.
const AUTO_EXPAND_MS = 2000;

// Terminal states that need no action linger just long enough to be read
// (#292). A failure or a stall stays until the user dismisses it.
const COMPLETED_DISMISS_MS = 4000;
const CANCELLED_DISMISS_MS = 2500;

const DownloadModalContext = createContext();

export function DownloadModalProvider({ children }) {
  const { t } = useTranslation();
  const [model, setModel] = useState(null);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  // `isDownloading` is the consumer-facing "a job is in flight" flag (sidebar
  // dimming, bug-report icon). The widget outlives it: it stays mounted through
  // the terminal state until dismissed, which `isWidgetVisible` tracks.
  const [isDownloading, setIsDownloading] = useState(false);
  const [isWidgetVisible, setIsWidgetVisible] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("idle");
  const [timeLeft, setTimeLeft] = useState(null);
  const [totalBytes, setTotalBytes] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");
  const [jobId, setJobId] = useState(null);
  // Completion exposed as observable context STATE, not only as a stored callback.
  // A per-download onComplete can be overwritten by another opener (the ref is a
  // singleton) or lost when the page that registered it unmounts mid-download.
  // A monotonic counter lets any mounted consumer react to "a download finished"
  // regardless of who started it or whether it remounted since (#205).
  const [completionCount, setCompletionCount] = useState(0);
  const [lastCompletedAt, setLastCompletedAt] = useState(null);

  const intervalRef = useRef(null);
  const expandTimerRef = useRef(null);
  const dismissTimerRef = useRef(null);
  const callbacksRef = useRef({ onComplete: null, onError: null });
  // Stall guard state (#315): the last (status, progress) pair seen and how many
  // consecutive polls have reported it unchanged while at 100%.
  const stallRef = useRef({ signature: null, ticks: 0 });
  // The job the poll is allowed to act on. Cleared on cancel so a poll reply
  // already in flight cannot resurrect a job the backend has just dropped.
  const activeJobRef = useRef(null);

  useEffect(
    () => () => {
      clearInterval(intervalRef.current);
      clearTimeout(expandTimerRef.current);
      clearTimeout(dismissTimerRef.current);
    },
    []
  );

  const toggleCollapse = useCallback(() => {
    setIsCollapsed((c) => !c);
  }, []);

  const dismissWidget = useCallback(() => {
    clearTimeout(dismissTimerRef.current);
    setIsWidgetVisible(false);
  }, []);

  // A terminal failure must be readable at once: the pill alone would only say
  // "Download failed" with no model, no detail and no Dismiss until the 2 s
  // auto-expand (or a click) came round.
  const revealPanel = useCallback(() => {
    clearTimeout(expandTimerRef.current);
    setIsCollapsed(false);
  }, []);

  const scheduleDismiss = useCallback((delayMs) => {
    clearTimeout(dismissTimerRef.current);
    dismissTimerRef.current = setTimeout(() => setIsWidgetVisible(false), delayMs);
  }, []);

  const open = useCallback((selectedModel, { onComplete, onError } = {}) => {
    setModel(selectedModel);
    callbacksRef.current = { onComplete, onError };
    setErrorMessage("");
    setIsConfirmOpen(true);
  }, []);

  const cancelConfirm = useCallback(() => setIsConfirmOpen(false), []);

  // Local settle shared by every cancel path: the acknowledged one, the
  // no-job-yet one, the endpoint-failed one and the poll's 404.
  const settleCancelled = useCallback(() => {
    activeJobRef.current = null;
    clearInterval(intervalRef.current);
    setIsDownloading(false);
    setProgress(0);
    setStatus(DOWNLOAD_CANCELLED);
    scheduleDismiss(CANCELLED_DISMISS_MS);
  }, [scheduleDismiss]);

  const checkDownloadStatus = useCallback(
    async (id) => {
      try {
        const res = await tracedFetch(`${API_BASE_URL}/llms/downloads/${id}/status`);
        if (activeJobRef.current !== id) {
          return;
        }
        if (!res.ok) {
          if (res.status === 404) {
            // The job no longer exists (most likely cancelled and cleaned up).
            settleCancelled();
            return;
          }
          throw new Error(`Server responded with ${res.status}: ${res.statusText}`);
        }
        const data = await res.json();
        setProgress(data.progress);
        setStatus(data.status);
        setTimeLeft(data.time_left);
        setTotalBytes(data.total_bytes ?? 0);

        const isTerminal =
          data.status === "completed" ||
          data.status === "failed" ||
          data.status === DOWNLOAD_CANCELLED;

        // Stall guard (#315). Only armed once the transfer is done: past 100% the
        // job is finalizing locally, so an unchanging reply means it is wedged,
        // not slow.
        if (!isTerminal && (data.progress ?? 0) >= 100) {
          const signature = `${data.status}:${Math.floor(data.progress ?? 0)}`;
          if (stallRef.current.signature === signature) {
            stallRef.current.ticks += 1;
          } else {
            stallRef.current = { signature, ticks: 1 };
          }
          if (stallRef.current.ticks >= FINALIZE_STALL_TICKS) {
            clearInterval(intervalRef.current);
            stallRef.current = { signature: null, ticks: 0 };
            log.error(
              `Download job ${id} stuck at 100% in status "${data.status}" for ` +
                `${FINALIZE_STALL_TICKS} polls; giving up on the status poll`
            );
            setIsDownloading(false);
            setStatus(DOWNLOAD_STALLED);
            setErrorMessage(downloadStalledMessage());
            revealPanel();
            callbacksRef.current.onError?.(DOWNLOAD_STALLED);
            return;
          }
        } else {
          stallRef.current = { signature: null, ticks: 0 };
        }

        if (isTerminal) {
          clearInterval(intervalRef.current);
          setIsDownloading(false);
          if (data.status === "completed") {
            setCompletionCount((c) => c + 1);
            setLastCompletedAt(Date.now());
            scheduleDismiss(COMPLETED_DISMISS_MS);
            callbacksRef.current.onComplete?.();
          } else if (data.status === DOWNLOAD_CANCELLED) {
            scheduleDismiss(CANCELLED_DISMISS_MS);
            callbacksRef.current.onError?.(DOWNLOAD_CANCELLED);
          } else {
            // The backend logged the traceback under this job; this side
            // records that the download the user started did not happen.
            log.error(`Download job ${id} failed with status "${data.status}"`, {
              error: data.error_message || null,
            });
            const errorMsg = data.error_message || t("downloads:errors.failedUnexpectedly");
            setErrorMessage(errorMsg);
            revealPanel();
            callbacksRef.current.onError?.(errorMsg);
          }
        }
      } catch (err) {
        log.error(`Status check failed for download job ${id}; polling stopped`, err);
        clearInterval(intervalRef.current);
        setIsDownloading(false);
        const errorMsg = t("downloads:errors.pollFailed");
        setErrorMessage(errorMsg);
        revealPanel();
        callbacksRef.current.onError?.(errorMsg);
      }
    },
    [t, scheduleDismiss, settleCancelled, revealPanel]
  );

  const handleConfirm = useCallback(async () => {
    setIsConfirmOpen(false);
    clearTimeout(dismissTimerRef.current);
    setIsWidgetVisible(true);
    setIsCollapsed(true);
    setIsDownloading(true);
    setStatus("pending");
    setProgress(0);
    setTimeLeft(null);
    setTotalBytes(0);
    setErrorMessage("");

    clearTimeout(expandTimerRef.current);
    expandTimerRef.current = setTimeout(() => setIsCollapsed(false), AUTO_EXPAND_MS);

    try {
      // Catalog models download by id; HF live-search hits have no id, so they
      // download by repo link via the dedicated endpoint (#122). Either way the
      // backend returns a DownloadJob we poll identically.
      const res =
        typeof model.id === "number"
          ? await tracedFetch(`${API_BASE_URL}/llms/${model.id}/download`, { method: "POST" })
          : await tracedFetch(`${API_BASE_URL}/llms/download/huggingface`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                link: model.link,
                name: model.name,
                type: model.type || null,
                // Preserve an unmeasured size as null instead of laundering it into
                // a plausible 7.0 (#201); the backend stores NULL = size unknown.
                param_size: model.param_size ?? null,
                quantized: model.quantized !== false,
                category: model.category || "general",
              }),
            });
      if (!res.ok) {
        const detail = downloadFailureDetail(await res.text());
        throw new Error(t("downloads:errors.startFailed", { status: res.status, detail }));
      }
      const job = await res.json();

      // Keep the job id around for cancellation.
      setJobId(job.id);
      activeJobRef.current = job.id;

      stallRef.current = { signature: null, ticks: 0 };
      intervalRef.current = setInterval(() => {
        checkDownloadStatus(job.id);
      }, POLL_INTERVAL_MS);
    } catch (err) {
      log.error("Download start error:", err);
      const errorMsg = err.message || err.toString() || t("downloads:errors.unexpected");
      setErrorMessage(errorMsg);
      revealPanel();
      setIsDownloading(false);
      callbacksRef.current.onError?.(errorMsg);
    }
  }, [model, checkDownloadStatus, t, revealPanel]);

  const cancelDownload = useCallback(async () => {
    if (!jobId) {
      // No job id yet: nothing to tell the backend, settle locally.
      settleCancelled();
      callbacksRef.current.onError?.(DOWNLOAD_CANCELLED);
      return;
    }

    try {
      const response = await tracedFetch(`${API_BASE_URL}/llms/downloads/${jobId}/cancel`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }

      log.log(`Download cancelled for job ${jobId}`);
    } catch (error) {
      log.error("Failed to cancel download:", error);
      setErrorMessage(t("downloads:errors.cancelFailed", { detail: error.message }));
    }

    // The backend drops the job on acknowledgement: one more status poll would
    // only be answered with a not-found warning in its log. Settle locally
    // right away -- and do the same when the cancel call itself failed, so the
    // user is never left with a widget they cannot get rid of.
    settleCancelled();
    callbacksRef.current.onError?.(DOWNLOAD_CANCELLED);
  }, [jobId, t, settleCancelled]);

  // What the widget renders, derived from the raw job state above. Bytes and
  // speed come from the status reply's `total_bytes` and `time_left`: the
  // backend's ETA is remaining / rate, so rate = remaining / ETA.
  const phase = deriveDownloadPhase({ status, progress, errorMessage });
  const downloadedBytes = totalBytes > 0 ? (totalBytes * (progress ?? 0)) / 100 : null;
  const speedBytesPerSec =
    phase === "downloading" && totalBytes > 0 && timeLeft > 0
      ? (totalBytes - downloadedBytes) / timeLeft
      : null;

  const portalRoot = typeof document === "undefined" ? null : document.getElementById("modal-root");

  return (
    <DownloadModalContext.Provider
      value={{
        open,
        isDownloading,
        completionCount,
        lastCompletedAt,
      }}
    >
      {children}

      {portalRoot &&
        ReactDOM.createPortal(
          <>
            {isConfirmOpen && (
              <ConfirmationModal
                isOpen
                onCancel={cancelConfirm}
                onConfirm={handleConfirm}
                text={model?.name}
              />
            )}
            <AnimatePresence>
              {isWidgetVisible && (
                <DownloadWidget
                  key="download-widget"
                  modelName={model?.name ?? ""}
                  phase={phase}
                  progress={progress ?? 0}
                  timeLeft={timeLeft}
                  totalBytes={totalBytes}
                  downloadedBytes={downloadedBytes}
                  speedBytesPerSec={speedBytesPerSec}
                  message={errorMessage}
                  collapsed={isCollapsed}
                  onToggleCollapse={toggleCollapse}
                  onCancel={cancelDownload}
                  onDismiss={dismissWidget}
                />
              )}
            </AnimatePresence>
          </>,
          portalRoot
        )}
    </DownloadModalContext.Provider>
  );
}

export function useDownloadModal() {
  return useContext(DownloadModalContext);
}
