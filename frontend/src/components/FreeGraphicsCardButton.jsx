import React, { useState } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import useModelUnload from "../shared/hooks/useModelUnload";

/**
 * The "Free the graphics card" action of the amber GPU contention notice
 * (another app is taking the card's memory, Windows CUDA). Pressing it unloads
 * the conversation's model (`useModelUnload`, the same request as the header's
 * Unload button); the next message reloads it, and that load places on the
 * card only what fits then. Erudi never does this on its own.
 *
 * Disabled while `disabled` holds (an answer is being written: the backend
 * would refuse, the engine is busy) and while the request runs. A refusal
 * (409: busy, or another model was loaded since) is said beside the action and
 * the notice stays; `disabled` changing (a turn started or ended) clears it.
 * Once the model is unloaded, `onFreed` is called (the page hides the notice).
 */
export default function FreeGraphicsCardButton({ llmId, disabled, onFreed }) {
  const { t } = useTranslation();
  const { unload, pending, refusal, setRefusal } = useModelUnload();

  // A turn started or ended: a refusal shown earlier no longer describes the
  // engine.
  const [seenDisabled, setSeenDisabled] = useState(disabled);
  if (seenDisabled !== disabled) {
    setSeenDisabled(disabled);
    setRefusal(null);
  }

  const free = async () => {
    if (await unload(llmId)) onFreed();
  };

  return (
    <span className="ml-auto inline-flex flex-shrink-0 flex-wrap items-center justify-end gap-2">
      {refusal && <span className="text-amber-100/90">{refusal}</span>}
      <button
        type="button"
        onClick={free}
        disabled={disabled || pending || llmId === null || llmId === undefined}
        className={[
          "rounded-md px-2.5 py-1 font-semibold whitespace-nowrap",
          "border border-amber-500/40 bg-amber-500/10 text-amber-100",
          "hover:border-amber-400/70 hover:bg-amber-500/20 transition",
          "disabled:opacity-50 disabled:cursor-not-allowed",
        ].join(" ")}
      >
        {t("chat:memoryWarning.freeGraphicsCard")}
      </button>
    </span>
  );
}

FreeGraphicsCardButton.propTypes = {
  // The conversation's model: the one the notice is about.
  llmId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  disabled: PropTypes.bool,
  onFreed: PropTypes.func.isRequired,
};
