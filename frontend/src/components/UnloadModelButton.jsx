import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import apiClient from "../services/api/client";

/**
 * The Unload button of the conversation header's settings panel (#619).
 *
 * Rendered as a child of `ConversationHeader`, so it exists only while the
 * panel is open: it asks the backend which model is loaded when it appears
 * (`GET /llms/loaded`) and names that model, whichever conversation loaded
 * it. Nothing loaded (or no answer): nothing is shown. Pressing it unloads
 * the model (`POST /llms/loaded/unload`) and asks again, so the button goes
 * once the model has left memory. While the engine is busy (an answer, a
 * reload, a memory check) the backend refuses with 409 and nothing is
 * interrupted; the button says so and stays. Request failures are logged by
 * the API client where the request is made.
 */
export default function UnloadModelButton() {
  const { t } = useTranslation();
  const [loaded, setLoaded] = useState(null);
  const [pending, setPending] = useState(false);
  const [refusal, setRefusal] = useState(null);
  const mounted = useRef(true);

  const refresh = useCallback(async () => {
    let next = null;
    try {
      const answer = await apiClient.get("/llms/loaded");
      next = answer && typeof answer.name === "string" ? answer : null;
    } catch {
      // Already logged by the API client; without an answer there is
      // nothing to offer to unload.
    }
    if (mounted.current) setLoaded(next);
  }, []);

  useEffect(() => {
    mounted.current = true;
    refresh();
    return () => {
      mounted.current = false;
    };
  }, [refresh]);

  const unload = async () => {
    setPending(true);
    setRefusal(null);
    try {
      await apiClient.post("/llms/loaded/unload");
    } catch (error) {
      // Logged by the API client; the user is told why nothing happened.
      if (mounted.current) {
        setRefusal(
          error?.status === 409
            ? t("chat:conversationHeader.unloadBusy")
            : t("chat:conversationHeader.unloadFailed")
        );
      }
    }
    await refresh();
    if (mounted.current) setPending(false);
  };

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
