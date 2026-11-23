import React from "react";
import PropTypes from "prop-types";
import { motion, AnimatePresence } from "framer-motion";
import { X, AlertTriangle } from "lucide-react";
import { useTranslation } from "react-i18next";

ErrorModal.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  message: PropTypes.string.isRequired,
  onClose: PropTypes.func.isRequired,
};

export default function ErrorModal({ errorMessage, onClose }) {
  const { t } = useTranslation();
  return (
    <AnimatePresence>
      {errorMessage && (
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
            className="relative w-full max-w-md"
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
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center">
                      <AlertTriangle className="w-4 h-4 text-red-400" />
                    </div>
                    <h2 className="text-xl font-semibold tracking-tight text-[#F2F7F4]">
                      {t("common:status.error")}
                    </h2>
                  </div>
                  <button
                    onClick={onClose}
                    className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 text-gray-300 hover:text-gray-100 transition"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* Error message */}
                <div className="mb-6">
                  <div className="bg-red-500/10 border border-red-400/20 rounded-2xl p-4">
                    {/* pre-line: a message made of several parts keeps its line breaks. */}
                    <p className="text-sm text-red-200 whitespace-pre-line">{errorMessage}</p>
                  </div>
                </div>

                {/* Button */}
                <div className="flex items-center justify-end">
                  <button
                    onClick={onClose}
                    className={[
                      "rounded-full px-5 py-2 text-sm font-semibold",
                      "bg-emerald-500 hover:bg-emerald-600 text-[#0f2f25]",
                      "border border-white/20 shadow",
                      "transition active:scale-95",
                    ].join(" ")}
                  >
                    {t("common:actions.close")}
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
