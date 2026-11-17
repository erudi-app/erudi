import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Database, Loader2, AlertTriangle } from "lucide-react";
import apiClient from "../services/api/client";
import { GATE, gateStateFromStatus, shouldPoll } from "../utils/embeddingGate";
import { createLogger } from "../utils/logger";

const log = createLogger("KbEmbeddingBanner");

const STATUS_PATH = "/knowledge_base/embedding-model/status";
const DOWNLOAD_PATH = "/knowledge_base/embedding-model/download";
const POLL_MS = 2000;

// "Knowledge Base present, embedding model missing" surfaced as a STATE in a
// conversation or an Arena panel (#594) — never a turn error. Shown only when
// `active` (the assistant has a Knowledge Base) and the backend says the
// model is absent: without it the assistant quietly answers "not in the
// documents" because its searches cannot embed. Reuses the #146 gate's
// reducer and endpoints; non-blocking (the user can still chat without the
// Knowledge Base), unlike the KB page's modal, which guards ingestion.
export default function KbEmbeddingBanner({ active }) {
  const { t } = useTranslation();
  const [state, setState] = useState(GATE.CHECKING);
  const [error, setError] = useState(null);
  const checkFailedRef = useRef(false);

  const refresh = async (prev) => {
    try {
      const status = await apiClient.get(STATUS_PATH, { silentFailure: true });
      checkFailedRef.current = false;
      setError(status.error || null);
      setState((current) => gateStateFromStatus(status, prev ?? current));
    } catch (err) {
      // One record per outage, not one per poll tick.
      if (!checkFailedRef.current) {
        checkFailedRef.current = true;
        log.error("Embedding-model status check failed", err);
      }
    }
  };

  useEffect(() => {
    if (!active) return undefined;
    // refresh sets state only once the request has answered (the rule does
    // not see the await): nothing is set while the effect runs.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refresh(GATE.CHECKING);
    return undefined;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  useEffect(() => {
    if (!active || !shouldPoll(state)) return undefined;
    const id = setInterval(() => refresh(), POLL_MS);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, state]);

  // The model can leave or reach the disk while the app is in the
  // background (a cleared models_cache, a download finished from the KB
  // page): coming back re-checks the state without a remount.
  useEffect(() => {
    if (!active) return undefined;
    const onFocus = () => refresh();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  const handleDownload = async () => {
    setState(GATE.DOWNLOADING);
    setError(null);
    try {
      await apiClient.post(DOWNLOAD_PATH);
    } catch (err) {
      // The download the user asked for did not start.
      log.error("Embedding-model download request failed", err);
      setError(String(err?.message || err));
      setState(GATE.ERROR);
    }
  };

  if (!active || state === GATE.CHECKING || state === GATE.HIDDEN || state === GATE.DONE) {
    return null;
  }

  if (state === GATE.DOWNLOADING) {
    return (
      <div
        role="status"
        className="mb-3 flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs"
      >
        <Loader2 className="w-4 h-4 flex-shrink-0 animate-spin" />
        <span>{t("downloads:embeddingGate.bannerDownloading")}</span>
      </div>
    );
  }

  const isError = state === GATE.ERROR;
  return (
    <div
      role="alert"
      className="mb-3 flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs"
    >
      <span className="flex items-center gap-2">
        {isError ? (
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
        ) : (
          <Database className="w-4 h-4 flex-shrink-0" />
        )}
        <span>
          {isError
            ? t("downloads:embeddingGate.bannerError")
            : t("downloads:embeddingGate.bannerMissing")}
          {isError && error ? <span className="block text-amber-200/60">{error}</span> : null}
        </span>
      </span>
      <button
        type="button"
        onClick={handleDownload}
        className="shrink-0 rounded-md px-2.5 py-1 font-semibold bg-white/10 hover:bg-white/15 border border-white/20 text-gray-100 transition"
      >
        {isError ? t("common:actions.retry") : t("common:actions.download")}
      </button>
    </div>
  );
}
