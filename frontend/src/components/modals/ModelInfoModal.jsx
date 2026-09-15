import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { X, Download, Users, Heart, Calendar, Tag, ChevronDown, BadgeCheck } from "lucide-react";
import { hasNoPublisherRecommendation } from "../../utils/samplingDefaults";
import { displayModelSize } from "../../utils/modelSize";

/**
 * Props:
 * - modelInfo: object with model information
 * - isOpen: boolean
 * - onClose: () => void
 * - onDownload: (modelInfo) => void
 * - installed: boolean — the model is already on disk (opened from an Installed
 *   card, or from a catalog card the local list joins to). No Download then.
 */

ModelInfoModal.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  model: PropTypes.shape({
    name: PropTypes.string,
    description: PropTypes.string,
    size: PropTypes.string,
    parameters: PropTypes.string,
    param_size: PropTypes.number,
    quantized: PropTypes.bool,
    artifact_size_bytes: PropTypes.number,
  }),
  onClose: PropTypes.func.isRequired,
  installed: PropTypes.bool,
};

ModelInfoModal.defaultProps = {
  model: null,
  installed: false,
};

// "Unknown" is the backend's sentinel for an absent field (also written by the
// Hugging Face search mapping): such a field is hidden, never shown as copy.
const isKnown = (value) => value && value !== "Unknown";

export default function ModelInfoModal({
  modelInfo,
  isOpen,
  onClose,
  onDownload,
  installed = false,
}) {
  const { t } = useTranslation();
  const [showRawMetadata, setShowRawMetadata] = useState(false);

  // Escape closes the modal, the way every dialog is expected to; the
  // listener only exists while the modal is open.
  useEffect(() => {
    if (!isOpen) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [isOpen, onClose]);

  return (
    <AnimatePresence>
      {isOpen && modelInfo && (
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
            className="relative w-full max-w-3xl max-h-[80vh] overflow-hidden"
          >
            {/* Modal container with HeaderBar-like styling */}
            <div
              className={[
                "relative w-full rounded-[26px] overflow-hidden",
                "border border-white/10",
                "bg-[rgba(22,40,36,0.45)] backdrop-blur-[18px] saturate-[1.4]",
                "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)]",
              ].join(" ")}
            >
              {/* Glossy overlays matching HeaderBar */}
              <div
                aria-hidden
                className="absolute inset-0 pointer-events-none rounded-[26px] mix-blend-overlay"
                style={{
                  background:
                    "linear-gradient(to bottom, rgba(255,255,255,0.18), rgba(255,255,255,0) 40%)",
                }}
              />

              {/* Content */}
              <div className="relative z-10 flex flex-col h-full max-h-[80vh]">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-white/10">
                  <div className="flex-1">
                    <h2 className="text-xl font-semibold tracking-tight text-[#F2F7F4] mb-1">
                      {modelInfo.name}
                    </h2>
                    <p className="text-sm text-gray-300/80">
                      {modelInfo.description || t("models:info.noDescription")}
                    </p>
                  </div>
                  <button
                    onClick={onClose}
                    className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 text-gray-300 hover:text-gray-100 transition ml-4"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* Scrollable content */}
                <div className="flex-1 overflow-y-auto p-6">
                  {/* Metadata Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold text-[#F2F7F4] flex items-center gap-2">
                        <Tag className="w-4 h-4 text-emerald-400" />
                        {t("models:info.basicInfo")}
                      </h3>
                      <div className="space-y-3 text-sm">
                        <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                          <span className="text-emerald-400 font-medium">
                            {t("models:info.size")}
                          </span>
                          <span className="text-gray-200 ml-2">
                            {displayModelSize(modelInfo) ?? modelInfo.size}
                          </span>
                        </div>
                        <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                          <span className="text-emerald-400 font-medium">
                            {t("models:info.parameters")}
                          </span>
                          <span className="text-gray-200 ml-2">{modelInfo.parameters}</span>
                        </div>
                        {/* Trained window = a fact of the model; allocated =
                            what the engine's loaded child runs with right
                            now, present only when this model is the loaded
                            one (the backend resolves it live). */}
                        {typeof modelInfo.context_window === "number" && (
                          <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                            <span className="text-emerald-400 font-medium">
                              {t("models:info.contextWindow")}
                            </span>
                            <span className="text-gray-200 ml-2">
                              {t("models:info.contextWindowTokens", {
                                tokens: modelInfo.context_window,
                              })}
                            </span>
                          </div>
                        )}
                        {typeof modelInfo.allocated_context_window === "number" && (
                          <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                            <span className="text-emerald-400 font-medium">
                              {t("models:info.allocatedContextWindow")}
                            </span>
                            <span className="text-gray-200 ml-2">
                              {t("models:info.contextWindowTokens", {
                                tokens: modelInfo.allocated_context_window,
                              })}
                            </span>
                          </div>
                        )}
                        {isKnown(modelInfo.author) && (
                          <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                            <span className="text-emerald-400 font-medium">
                              {t("models:info.author")}
                            </span>
                            <span className="text-gray-200 ml-2">{modelInfo.author}</span>
                          </div>
                        )}
                        {isKnown(modelInfo.library) && (
                          <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                            <span className="text-emerald-400 font-medium">
                              {t("models:info.library")}
                            </span>
                            <span className="text-gray-200 ml-2">{modelInfo.library}</span>
                          </div>
                        )}
                        {hasNoPublisherRecommendation(modelInfo) && (
                          <p
                            data-testid="no-publisher-recommendation"
                            className="text-xs leading-snug text-gray-400/80 px-1"
                          >
                            {t("models:info.noPublisherRecommendation")}
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold text-[#F2F7F4] flex items-center gap-2">
                        <Users className="w-4 h-4 text-emerald-400" />
                        {t("models:info.stats")}
                      </h3>
                      <div className="space-y-3 text-sm">
                        {isKnown(modelInfo.downloads) && (
                          <div className="bg-white/5 rounded-xl p-3 border border-white/10 flex items-center gap-2">
                            <Download className="w-4 h-4 text-emerald-400" />
                            <span className="text-emerald-400 font-medium">
                              {t("models:info.downloads")}
                            </span>
                            <span className="text-gray-200">{modelInfo.downloads}</span>
                          </div>
                        )}
                        {isKnown(modelInfo.likes) && (
                          <div className="bg-white/5 rounded-xl p-3 border border-white/10 flex items-center gap-2">
                            <Heart className="w-4 h-4 text-emerald-400" />
                            <span className="text-emerald-400 font-medium">
                              {t("models:info.likes")}
                            </span>
                            <span className="text-gray-200">{modelInfo.likes}</span>
                          </div>
                        )}
                        {isKnown(modelInfo.lastUpdate) && (
                          <div className="bg-white/5 rounded-xl p-3 border border-white/10 flex items-center gap-2">
                            <Calendar className="w-4 h-4 text-emerald-400" />
                            <span className="text-emerald-400 font-medium">
                              {t("models:info.lastUpdate")}
                            </span>
                            <span className="text-gray-200">{modelInfo.lastUpdate}</span>
                          </div>
                        )}
                        {isKnown(modelInfo.pipeline) && (
                          <div className="bg-white/5 rounded-xl p-3 border border-white/10">
                            <span className="text-emerald-400 font-medium">
                              {t("models:info.pipeline")}
                            </span>
                            <span className="text-gray-200 ml-2">{modelInfo.pipeline}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Raw Metadata (collapsible) */}
                  {modelInfo.rawMetadata && (
                    <div className="border-t border-white/10 pt-6">
                      <button
                        onClick={() => setShowRawMetadata(!showRawMetadata)}
                        className="flex items-center gap-2 cursor-pointer text-[#F2F7F4] font-medium mb-3 hover:text-emerald-400 transition-colors select-none group"
                      >
                        <motion.div
                          animate={{ rotate: showRawMetadata ? 180 : 0 }}
                          transition={{ duration: 0.2 }}
                        >
                          <ChevronDown className="w-4 h-4" />
                        </motion.div>
                        {t("models:info.showRawMetadata")}
                      </button>

                      <AnimatePresence>
                        {showRawMetadata && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{
                              duration: 0.3,
                              ease: [0.16, 1, 0.3, 1],
                            }}
                            style={{ overflow: "hidden" }}
                          >
                            <motion.div
                              initial={{ y: -10 }}
                              animate={{ y: 0 }}
                              exit={{ y: -10 }}
                              transition={{
                                duration: 0.3,
                                ease: [0.16, 1, 0.3, 1],
                              }}
                              className="bg-black/40 border border-white/10 p-4 rounded-2xl text-xs text-gray-300 whitespace-pre-wrap max-h-60 overflow-auto"
                            >
                              {modelInfo.rawMetadata}
                            </motion.div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  )}
                </div>

                {/* Footer. An installed model must not offer its own download
                    again: the state is shown instead and the only action is Close. */}
                <div className="flex items-center justify-end gap-3 p-6 border-t border-white/10">
                  {installed ? (
                    <>
                      <span className="inline-flex items-center gap-1.5 text-sm font-medium text-emerald-300/90">
                        <BadgeCheck className="w-4 h-4" />
                        {t("models:info.installed")}
                      </span>
                      <button
                        onClick={onClose}
                        className={[
                          "rounded-full px-5 py-2 text-sm font-semibold",
                          "bg-white/10 hover:bg-white/15 text-gray-100",
                          "border border-white/20 backdrop-blur-sm shadow-sm",
                          "transition active:scale-95",
                        ].join(" ")}
                      >
                        {t("common:actions.close")}
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={onClose}
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
                        onClick={() => {
                          onDownload(modelInfo);
                          onClose();
                        }}
                        className={[
                          "rounded-full px-5 py-2 text-sm font-semibold",
                          "bg-emerald-500 hover:bg-emerald-600 text-[#0f2f25]",
                          "border border-white/20 shadow",
                          "transition active:scale-95",
                          "flex items-center gap-2",
                        ].join(" ")}
                      >
                        <Download className="w-4 h-4" />
                        {t("common:actions.download")}
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
