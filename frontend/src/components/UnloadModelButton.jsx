import React, { useCallback, useEffect, useRef, useState } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import apiClient from "../services/api/client";
import useModelUnload from "../shared/hooks/useModelUnload";

/**
 * The Unload button of the conversation header's settings panel (#619).
 *
 * Rendered as a child of `ConversationHeader`, so it exists only while the
 * panel is open: it asks the backend which model is loaded when it appears
 * (`GET /llms/loaded`), again whenever the page's `refreshKey` changes (the
 * conversation page changes it when a turn has completed, after its own
 * refetch, and when the conversation's model changes; a change also clears a
 * refusal shown earlier), and names that model, whichever conversation loaded
 * it. Nothing loaded (or no answer): nothing is
 * shown. Pressing it unloads the model it names (`POST /llms/loaded/unload`
 * with its `llm_id`, through `useModelUnload`) and asks again, so the button
 * goes once the model has left memory. A refusal (409: the engine is busy, or
 * another model was loaded since the button read it) is said beside it, and
 * the loaded model is read again. Only the latest read is shown. Request
 * failures are logged by the API client where the request is made.
 */
export default function UnloadModelButton({ refreshKey }) {
  const { t } = useTranslation();
  const [loaded, setLoaded] = useState(null);
  const { unload: unloadModel, pending, refusal, setRefusal } = useModelUnload();
  const mounted = useRef(true);
  const latestRead = useRef(0);

  const refresh = useCallback(async () => {
    latestRead.current += 1;
    const read = latestRead.current;
    let next = null;
    try {
      const answer = await apiClient.get("/llms/loaded");
      next = answer && typeof answer.name === "string" ? answer : null;
    } catch {
      // Already logged by the API client; without an answer there is
      // nothing to offer to unload.
    }
    if (mounted.current && read === latestRead.current) setLoaded(next);
  }, []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // Something changed (a turn started or ended, the model changed): a refusal
  // shown earlier no longer describes the engine.
  const [seenRefreshKey, setSeenRefreshKey] = useState(refreshKey);
  if (seenRefreshKey !== refreshKey) {
    setSeenRefreshKey(refreshKey);
    setRefusal(null);
  }

  useEffect(() => {
    refresh();
  }, [refresh, refreshKey]);

  const unload = () => unloadModel(loaded.llm_id, refresh);

  if (!loaded) return null;

  return (
    <span className="ml-auto inline-flex flex-wrap items-center justify-end gap-2">
      {refusal && (
        <span role="alert" className="text-xs text-red-300/90">
          {refusal}
        </span>
      )}
      <button
        type="button"
        onClick={unload}
        disabled={pending}
        title={loaded.name}
        className={[
          "inline-flex items-center rounded-lg px-2.5 py-1 text-xs font-medium",
          "border border-red-500/30 bg-red-500/5 text-red-300/90",
          "hover:border-red-400/60 hover:bg-red-500/10 hover:text-red-200 transition",
          "disabled:opacity-50 disabled:cursor-not-allowed",
        ].join(" ")}
      >
        {t("chat:conversationHeader.unload", { model: loaded.name })}
      </button>
    </span>
  );
}

UnloadModelButton.propTypes = {
  // Any value the page changes when the loaded model may have changed.
  refreshKey: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
};
