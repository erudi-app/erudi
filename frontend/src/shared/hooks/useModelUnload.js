import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import apiClient from "../../services/api/client";

const UNLOAD_PATH = "/llms/loaded/unload";

/** The copy of a refused unload: busy, another model loaded, or a failure. */
function refusalKey(error) {
  if (error?.status !== 409) return "chat:conversationHeader.unloadFailed";
  return error.data?.error?.type === "LOADED_MODEL_CHANGED"
    ? "chat:conversationHeader.unloadModelChanged"
    : "chat:conversationHeader.unloadBusy";
}

/**
 * Unloading the loaded model on the user's request (`POST /llms/loaded/unload`
 * with its `llm_id`), shared by the conversation header's Unload button and
 * the GPU contention notice's "Free the graphics card" action.
 *
 * `unload(llmId, after)` sends the request, then awaits `after` (the caller's
 * own follow-up, such as reading the loaded model again) before `pending`
 * goes back to false; it resolves to whether the model was unloaded. The
 * backend refuses with 409 while the engine is busy (an answer, a reload, a
 * memory check; nothing is interrupted) or when another model was loaded
 * since (nothing is unloaded): `refusal` then holds the sentence that says
 * which, or the generic failure for any other error. Request failures are
 * logged by the API client where the request is made.
 */
export default function useModelUnload() {
  const { t } = useTranslation();
  const [pending, setPending] = useState(false);
  const [refusal, setRefusal] = useState(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const unload = useCallback(
    async (llmId, after) => {
      setPending(true);
      setRefusal(null);
      let unloaded = false;
      try {
        await apiClient.post(UNLOAD_PATH, { llm_id: llmId });
        unloaded = true;
      } catch (error) {
        // Logged by the API client; the user is told why nothing happened.
        if (mounted.current) setRefusal(t(refusalKey(error)));
      }
      if (after) await after(unloaded);
      if (mounted.current) setPending(false);
      return unloaded;
    },
    [t]
  );

  return { unload, pending, refusal, setRefusal };
}
