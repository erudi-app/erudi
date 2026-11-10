import React, { useState } from "react";
import PropTypes from "prop-types";
import { motion, AnimatePresence } from "framer-motion";
import { useTranslation } from "react-i18next";

CustomizePromptModal.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  onSave: PropTypes.func.isRequired,
  initialPrompt: PropTypes.string,
};

CustomizePromptModal.defaultProps = {
  initialPrompt: "",
};

export default function CustomizePromptModal({ isOpen, onClose, customPrompt, onSave, title }) {
  const { t } = useTranslation();
  const heading = title ?? t("chat:prompt.title");
  const [localPrompt, setLocalPrompt] = useState(customPrompt);

  // Update local state when the prop changes (when switching conversations)
  const [seenPrompt, setSeenPrompt] = useState(customPrompt);
  if (seenPrompt !== customPrompt) {
    setSeenPrompt(customPrompt);
    setLocalPrompt(customPrompt);
  }

  const handleSave = () => {
    onSave?.(localPrompt);
    onClose();
  };

  const handleCancel = () => {
    // Reset to original value on cancel
    setLocalPrompt(customPrompt);
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[9999] p-4"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="relative w-full max-w-2xl"
          >
            <div
              className={[
                "relative w-full rounded-[26px] overflow-hidden",
                "border border-white/10",
                "bg-[rgba(22,40,36,0.45)] backdrop-blur-[18px] saturate-[1.4]",
                "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)]",
              ].join(" ")}
            >
              <div
                aria-hidden
                className="absolute inset-0 pointer-events-none rounded-[26px] mix-blend-overlay"
                style={{
                  background:
                    "linear-gradient(to bottom, rgba(255,255,255,0.18), rgba(255,255,255,0) 40%)",
                }}
              />

              {/* Content */}
              <div className="relative z-10 p-6">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-semibold tracking-tight text-[#F2F7F4]">{heading}</h2>
                  <button
                    onClick={onClose}
                    className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 text-gray-300 hover:text-gray-100 transition"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>
                </div>

                {/* Description */}
                <p className="text-sm text-gray-300/80 mb-4">{t("chat:prompt.description")}</p>

                {/* Textarea */}
                <div className="mb-6">
                  <textarea
                    className={[
                      "w-full h-40 rounded-2xl resize-none",
                      "bg-white/10 border border-white/20",
                      "px-4 py-3 text-sm text-gray-100 placeholder-gray-400",
                      "focus:outline-none focus:ring-2 focus:ring-emerald-400/60 focus:border-emerald-400/60",
                      "backdrop-blur-sm transition-all custom-scroll",
                    ].join(" ")}
                    value={localPrompt}
                    onChange={(e) => setLocalPrompt(e.target.value)}
                    placeholder={t("chat:prompt.placeholder")}
                  />
                </div>

                {/* Buttons */}
                <div className="flex items-center justify-end gap-3">
                  <button
                    onClick={handleCancel}
                    className={[
                      "rounded-full px-5 py-2 text-sm font-semibold",
                      "bg-white/10 hover:bg-white/15 text-gray-100",
                      "border border-white/20 backdrop-blur-sm shadow-sm",
                      "transition active:scale-95",
                    ].join(" ")}
                  >
                    {t("common:actions.cancel")}
                  </button>
                  <button
                    onClick={handleSave}
                    className={[
                      "rounded-full px-5 py-2 text-sm font-semibold",
                      "bg-emerald-500 hover:bg-emerald-600 text-[#0f2f25]",
                      "border border-white/20 shadow",
                      "transition active:scale-95",
                    ].join(" ")}
                  >
                    {t("chat:prompt.saveChanges")}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
