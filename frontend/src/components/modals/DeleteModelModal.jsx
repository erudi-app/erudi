import React from "react";
import PropTypes from "prop-types";
import { Trans, useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { X, Trash2 } from "lucide-react";
import { displayModelSize } from "../../utils/modelSize";

DeleteModelModal.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  model: PropTypes.shape({
    name: PropTypes.string,
    size: PropTypes.string,
    param_size: PropTypes.number,
    quantized: PropTypes.bool,
    artifact_size_bytes: PropTypes.number,
  }),
  onConfirm: PropTypes.func.isRequired,
  onCancel: PropTypes.func.isRequired,
  // Guarded base delete (#225): dependents of the model about to be deleted,
  // in the shape of GET /llms/{id}/dependents. When present with at least one
  // assistant, the dialog explains the consequences and the confirm button
  // becomes "Delete anyway" (the DELETE then orphans the dependents).
  dependents: PropTypes.shape({
    assistants: PropTypes.arrayOf(
      PropTypes.shape({
        id: PropTypes.number,
        name: PropTypes.string,
      })
    ),
    own_conversation_count: PropTypes.number,
    total_conversation_count: PropTypes.number,
  }),
};

export default function DeleteModelModal({ isOpen, model, onConfirm, onCancel, dependents }) {
  const { t } = useTranslation();
  const assistants = dependents?.assistants || [];
  const hasDependents = assistants.length > 0;
  const conversationCount = dependents?.total_conversation_count ?? 0;
  const sizeLabel = displayModelSize(model);
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
            className="relative w-full max-w-md overflow-hidden"
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
              <div className="relative z-10 flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-white/10">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center">
                      <Trash2 className="w-4 h-4 text-red-400" />
                    </div>
                    <h2 className="text-xl font-semibold tracking-tight text-[#F2F7F4]">
                      {t("downloads:delete.title")}
                    </h2>
                  </div>
                  <button
                    onClick={onCancel}
                    className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 text-gray-300 hover:text-gray-100 transition"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* Content */}
                <div className="p-6">
                  <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
                    {hasDependents ? (
                      <>
                        <p className="text-[#F2F7F4] text-sm">
                          <Trans
                            i18nKey="downloads:delete.dependentsSummary"
                            values={{
                              name: model?.name,
                              assistants: t("downloads:delete.assistantCount", {
                                count: assistants.length,
                              }),
                              names: assistants.map((a) => a.name).join(", "),
                              conversations: t("downloads:delete.conversationCount", {
                                count: conversationCount,
                              }),
                            }}
                            components={{
                              name: <span className="font-semibold text-red-400" />,
                              strong: <span className="font-semibold" />,
                            }}
                          />
                          {sizeLabel ? ` ${t("downloads:delete.frees", { size: sizeLabel })}` : ""}
                        </p>
                        <p className="text-gray-300/80 text-xs mt-2">
                          {t("downloads:delete.dependentsNote")}
                        </p>
                      </>
                    ) : (
                      <>
                        <p className="text-[#F2F7F4] text-sm">
                          <Trans
                            i18nKey="downloads:delete.confirmQuestion"
                            values={{ name: model?.name }}
                            components={{ name: <span className="font-semibold text-red-400" /> }}
                          />
                        </p>
                        <p className="text-gray-300/80 text-xs mt-2">
                          {t("downloads:delete.irreversible")}
                        </p>
                      </>
                    )}
                  </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end gap-3 p-6 border-t border-white/10">
                  <button
                    onClick={onCancel}
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
                    onClick={onConfirm}
                    className={[
                      "rounded-full px-5 py-2 text-sm font-semibold",
                      "bg-red-500 hover:bg-red-600 text-white",
                      "border border-white/20 shadow",
                      "transition active:scale-95",
                      "flex items-center gap-2",
                    ].join(" ")}
                  >
                    <Trash2 className="w-4 h-4" />
                    {hasDependents
                      ? t("downloads:delete.deleteAnyway")
                      : t("common:actions.delete")}
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
