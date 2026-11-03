import React, { useId, useState } from "react";
import PropTypes from "prop-types";
import { motion, AnimatePresence } from "framer-motion";
import { useTranslation } from "react-i18next";

/**
 * The question asked before switching the inference engine while a model
 * download is in progress: the switch restarts the engine, which cancels the
 * download, and the download then starts over from the beginning.
 *
 * `EngineSwitchConfirmationContent` is the question alone, rendered as a step
 * inside a dialog that is already open (`EngineFailureModal`); the default
 * export wraps it in a dialog of its own (the Settings page). The copy lives
 * in the `common` namespace because both share it.
 */
export function EngineSwitchConfirmationContent({ titleId, onConfirm, onCancel }) {
  const { t } = useTranslation();
  // The question is answered once. The dialog stays mounted through its exit
  // animation, so without this a second click would answer it again.
  const [answered, setAnswered] = useState(false);
  const answer = (callback) => () => {
    if (answered) return;
    setAnswered(true);
    callback();
  };
  return (
    <>
      <h2 id={titleId} className="text-xl font-semibold tracking-tight text-[#F2F7F4] mb-2">
        {t("common:engineSwitch.downloadTitle")}
      </h2>
      <p className="text-sm text-gray-300/80 mb-6">{t("common:engineSwitch.downloadBody")}</p>
      <div className="flex flex-wrap items-center justify-end gap-3">
        <button
          type="button"
          onClick={answer(onCancel)}
          disabled={answered}
          className="rounded-full px-5 py-2 text-sm font-semibold bg-white/10 hover:bg-white/15 text-gray-100 border border-white/20 transition active:scale-95"
        >
          {t("common:engineSwitch.keepDownloading")}
        </button>
        <button
          type="button"
          onClick={answer(onConfirm)}
          disabled={answered}
          className="rounded-full px-5 py-2 text-sm font-semibold bg-emerald-500 hover:bg-emerald-600 text-[#0f2f25] border border-white/20 shadow transition active:scale-95"
        >
          {t("common:engineSwitch.cancelAndSwitch")}
        </button>
      </div>
    </>
  );
}

EngineSwitchConfirmationContent.propTypes = {
  titleId: PropTypes.string,
  onConfirm: PropTypes.func.isRequired,
  onCancel: PropTypes.func.isRequired,
};

export default function EngineSwitchConfirmation({ isOpen, onConfirm, onCancel }) {
  const titleId = useId();
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          // Fading out, the dialog takes no more clicks (pointer-events is
          // not animated: framer-motion applies it as the exit starts).
          exit={{ opacity: 0, pointerEvents: "none" }}
          transition={{ duration: 0.2 }}
          role="dialog"
          aria-modal={true}
          aria-labelledby={titleId}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[9999] p-4"
        >
          <div className="relative w-full max-w-md rounded-[26px] border border-white/10 bg-[rgba(22,40,36,0.85)] backdrop-blur-[18px] shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45)] p-6">
            <EngineSwitchConfirmationContent
              titleId={titleId}
              onConfirm={onConfirm}
              onCancel={onCancel}
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

EngineSwitchConfirmation.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onConfirm: PropTypes.func.isRequired,
  onCancel: PropTypes.func.isRequired,
};
