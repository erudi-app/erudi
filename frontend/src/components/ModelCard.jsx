import React, { useEffect, useId, useRef, useState } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import GradientBox from "./GradientBox";
import {
  AlertTriangle,
  BadgeCheck,
  Download,
  Info,
  BookOpen,
  MessageSquare,
  Plus,
  Trash2,
  Link2,
} from "lucide-react";
import { useDownloadModal } from "../contexts/DownloadModalContext";
import { isKbAssistant, hasMissingWeights } from "../utils/modelWeights";
import { isTestedModel } from "../utils/testedModels";
import { isVerySmallModel, SMALL_MODEL_PARAM_THRESHOLD_B } from "../utils/modelCapabilities";
import { displayModelSize } from "../utils/modelSize";
import { publishedLabel } from "../utils/publishedDate";
import { createLogger } from "../utils/logger";

const log = createLogger("ModelCard");

/**
 * ModelCard component - displays model information with actions
 * @param {Object} model - Model object with name, size, description, etc.
 * @param {string} type - Card type: "base" (available), "local" (installed), "add" (add new)
 * @param {Function} onDownload - Callback when download button clicked
 * @param {Function} onInfo - Callback when info button clicked
 * @param {Function} onChat - Callback when chat button clicked
 * @param {Function} onKnowledgeBase - Callback when knowledge base button clicked
 * @param {Function} onDelete - Callback when delete button clicked
 * @param {Function} onClick - Callback when card clicked
 * @param {string} baseModelName - KB assistants: name of the installed base whose weights it uses
 * @param {Array} rebindTargets - Installed base models an orphaned assistant can be re-bound to
 * @param {Function} onRebind - Callback (assistant, target) when a re-bind target is picked
 */
function ModelCard({
  model,
  type = "base",
  onDownload,
  onInfo,
  onChat,
  onKnowledgeBase,
  onDelete,
  onClick: _onClick,
  baseModelName = null,
  rebindTargets = [],
  onRebind,
}) {
  const { t } = useTranslation();
  const publishedText = publishedLabel(model?.publishedAt);
  const { open } = useDownloadModal();
  const [rebindOpen, setRebindOpen] = useState(false);
  const rebindTriggerRef = useRef(null);
  const rebindRootRef = useRef(null);
  const rebindListId = useId();
  // Escape closes the re-bind menu and gives the focus back to its trigger.
  const closeRebind = (refocus) => {
    setRebindOpen(false);
    if (refocus) rebindTriggerRef.current?.focus();
  };
  useEffect(() => {
    if (!rebindOpen) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape" && rebindRootRef.current?.contains(document.activeElement)) {
        e.stopPropagation();
        closeRebind(true);
      }
    };
    const onMouseDown = (e) => {
      if (!rebindRootRef.current?.contains(e.target)) closeRebind(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onMouseDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onMouseDown);
    };
  }, [rebindOpen]);
  const unavailable = model?.runnable === false;
  // Orphan-model UX (#225/#208): a KB assistant uses its base model's weights
  // (its own link is a copy — it owns no disk space). When those weights are
  // gone, Chat is blocked (it can only fail) and a re-bind picker of installed
  // base models appears.
  const isAssistant = isKbAssistant(model);
  const weightsMissing = type === "local" && hasMissingWeights(model);
  const verySmall = isVerySmallModel(model);
  const handleDownload = () => {
    if (model) {
      open(model, {
        onComplete: () => onDownload && onDownload(model),
        // A failure comes as { message, code, error_message }; a cancel or a
        // stall as its status string, which the download context records
        // itself. The record names the code (the bug counter reads it) and
        // carries the backend's English message, not the translated one.
        onError: (failure) => {
          if (typeof failure !== "object" || failure === null) return;
          log.error(`Download failed for ${model.name} (code=${failure.code ?? "none"})`, {
            error: failure.error_message ?? null,
          });
        },
      });
    } else if (type === "add" && onDownload) {
      onDownload();
    }
  };

  if (type === "add") {
    return (
      <GradientBox
        className="bg-[#1a1a1a]/60 backdrop-blur-sm border border-white/10 border-dashed cursor-pointer hover:border-white/30 transition-colors"
        onClick={handleDownload}
      >
        <div className="flex flex-col items-center justify-center h-full text-center min-h-[160px]">
          <Plus className="w-8 h-8 text-white/60 mb-3" />
          <p className="text-white/60 text-sm">{t("models:card.addTitle")}</p>
          <p className="text-white/40 text-xs mt-1">{t("models:card.addSubtitle")}</p>
        </div>
      </GradientBox>
    );
  }

  return (
    <GradientBox className="bg-[#1a1a1a]/60 backdrop-blur-sm border border-white/10">
      <div className="flex flex-col h-full">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-1.5 min-w-0">
            <h3 className="min-w-0 truncate text-lg font-semibold text-white" title={model.name}>
              {model.name}
            </h3>
            {isTestedModel(model) && (
              <span
                className="flex items-center shrink-0 text-emerald-400"
                title={t("models:card.testedBadge")}
              >
                <BadgeCheck className="w-4 h-4" />
              </span>
            )}
          </div>
          {unavailable && type !== "local" && (
            <span className="ml-2 flex-shrink-0 text-[10px] uppercase tracking-wide text-amber-500/80 border border-amber-500/30 rounded px-1.5 py-0.5">
              {t("models:card.unavailable")}
            </span>
          )}
          {weightsMissing && (
            <span className="ml-2 flex-shrink-0 text-[10px] uppercase tracking-wide text-red-400/90 border border-red-500/40 rounded px-1.5 py-0.5">
              {t("models:card.weightsMissingBadge")}
            </span>
          )}
          {type === "local" && (
            <button
              className="p-1 bg-red-500/20 hover:bg-red-500/40 rounded-lg transition-colors ml-2"
              onClick={(e) => {
                e.stopPropagation();
                onDelete && onDelete(model);
              }}
              title={t("models:card.deleteModel")}
              aria-label={t("models:card.deleteModel")}
            >
              <Trash2 aria-hidden="true" className="w-4 h-4 text-red-400" />
            </button>
          )}
        </div>

        <div className="space-y-1 text-xs text-gray-300 mb-4">
          {/* Show description if available */}
          {model.description && (
            <p className="text-blue-300 font-medium mb-2 text-xs">{model.description}</p>
          )}

          {/* Core metadata. Assistant cards own no disk space — they use the
              base model's weights (#208), so no Size line for them. */}
          {type === "local" && isAssistant ? (
            weightsMissing ? (
              <p className="text-red-400 font-medium">{t("models:card.weightsMissing")}</p>
            ) : (
              <p>
                {t("models:card.usesWeightsOf", {
                  name: baseModelName || t("models:card.itsBaseModel"),
                })}
              </p>
            )
          ) : (
            <p>{t("models:card.size", { size: displayModelSize(model) ?? model.size })}</p>
          )}
          {/* Nothing at all when the publication date is unknown (#561). */}
          {!(type === "local" && isAssistant) && publishedText && <p>{publishedText}</p>}

          {/* Additional metadata for remote models. "Unknown" is the backend's
              sentinel for an absent value, not copy — it is compared, never shown. */}
          {type !== "local" && (
            <>
              {model.downloads && model.downloads !== "Unknown" && (
                <p>{t("models:card.downloads", { value: model.downloads })}</p>
              )}
              {model.likes && model.likes !== "Unknown" && (
                <p>{t("models:card.likes", { value: model.likes })}</p>
              )}
              {model.author && model.author !== "Unknown" && (
                <p>{t("models:card.author", { author: model.author })}</p>
              )}
              {model.library && model.library !== "Unknown" && (
                <p>{t("models:card.library", { library: model.library })}</p>
              )}
            </>
          )}

          {/* Local model specific info */}
          {type === "local" && model.lastUpdate && model.lastUpdate !== "Unknown" && (
            <>
              <p>{t("models:card.parameters", { parameters: model.parameters })}</p>
              <p>{t("models:card.lastUpdate", { date: model.lastUpdate })}</p>
            </>
          )}

          {/* Very small models (#381): warn on the card that tools, KB search
              and multi-step reasoning are unreliable, so a silent no-tool-call
              later does not read as a broken feature. */}
          {verySmall && (
            <p
              data-testid="small-model-note"
              className="flex items-start gap-1.5 pt-1 text-[11px] leading-snug text-gray-400"
            >
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-px text-amber-500/80" />
              <span>
                {t("models:smallModelNote", { threshold: SMALL_MODEL_PARAM_THRESHOLD_B })}
              </span>
            </p>
          )}
        </div>

        <div className="flex items-center gap-2 mt-auto">
          {type === "local" ? (
            <>
              <button
                className="p-1 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
                onClick={() => onKnowledgeBase && onKnowledgeBase(model)}
                title={t("common:nav.knowledgeBase")}
              >
                <BookOpen className="w-4 h-4 text-white" />
              </button>
              <button
                className={`p-1 rounded-lg transition-colors ${weightsMissing ? "bg-white/5 opacity-40 cursor-not-allowed" : "bg-white/10 hover:bg-white/20"}`}
                onClick={() => !weightsMissing && onChat && onChat(model)}
                disabled={weightsMissing}
                title={weightsMissing ? t("models:card.chatBlocked") : t("common:nav.chat")}
              >
                <MessageSquare className="w-4 h-4 text-white" />
              </button>
              <button
                className="p-1 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
                onClick={() => onInfo && onInfo(model)}
                title={t("models:card.info")}
              >
                <Info className="w-4 h-4 text-white" />
              </button>
              {isAssistant && weightsMissing && (
                <div
                  ref={rebindRootRef}
                  className="relative ml-auto"
                  onBlur={(e) => {
                    // Focus leaving the trigger and its list (Tab past the last
                    // entry) closes the list.
                    if (rebindOpen && !e.currentTarget.contains(e.relatedTarget))
                      closeRebind(false);
                  }}
                >
                  <button
                    ref={rebindTriggerRef}
                    type="button"
                    aria-controls={rebindOpen ? rebindListId : undefined}
                    aria-expanded={rebindOpen}
                    className="inline-flex focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60 items-center gap-1 px-2 py-1 text-xs font-semibold bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 rounded-lg transition-colors"
                    onClick={() => setRebindOpen((o) => !o)}
                    title={t("models:card.rebindTitle")}
                  >
                    <Link2 aria-hidden="true" className="w-3.5 h-3.5" />
                    {t("models:card.rebind")}
                  </button>
                  {rebindOpen && (
                    <div
                      id={rebindListId}
                      className="absolute bottom-full right-0 mb-1 w-48 bg-[#2a2a2a] border border-white/20 rounded-lg shadow-lg z-20 max-h-40 overflow-y-auto"
                    >
                      {rebindTargets.length === 0 ? (
                        <p className="px-3 py-2 text-xs text-gray-400">
                          {t("models:card.noRebindTarget")}
                        </p>
                      ) : (
                        rebindTargets.map((target) => (
                          <button
                            key={target.id}
                            type="button"
                            className="block w-full px-3 py-2 text-left text-xs hover:bg-white/10 focus:outline-none focus-visible:bg-white/10 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-emerald-400/60 text-gray-100 border-b border-white/10 last:border-b-0"
                            onClick={() => {
                              setRebindOpen(false);
                              onRebind && onRebind(model, target);
                            }}
                          >
                            {target.name}
                          </button>
                        ))
                      )}
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <>
              <button
                className={`p-1 rounded-lg transition-colors ${unavailable ? "bg-white/5 opacity-40 cursor-not-allowed" : "bg-white/10 hover:bg-white/20"}`}
                onClick={() => !unavailable && onDownload && onDownload(model)}
                disabled={unavailable}
                title={unavailable ? t("models:card.unavailable") : t("common:actions.download")}
              >
                <Download className="w-5 h-5 text-white" />
              </button>
              <button
                className="p-1 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
                onClick={() => onInfo && onInfo(model)}
                title={t("models:card.info")}
              >
                <Info className="w-5 h-5 text-white" />
              </button>
            </>
          )}
        </div>
      </div>
    </GradientBox>
  );
}

ModelCard.propTypes = {
  model: PropTypes.shape({
    name: PropTypes.string.isRequired,
    size: PropTypes.string,
    description: PropTypes.string,
    downloads: PropTypes.string,
    likes: PropTypes.string,
    publishedAt: PropTypes.string,
    author: PropTypes.string,
    library: PropTypes.string,
    parameters: PropTypes.string,
    param_size: PropTypes.number,
    quantized: PropTypes.bool,
    artifact_size_bytes: PropTypes.number,
    lastUpdate: PropTypes.string,
    runnable: PropTypes.bool,
    link: PropTypes.string,
    kb_id: PropTypes.number,
    is_attached_to_kb: PropTypes.bool,
    weights_available: PropTypes.bool,
  }).isRequired,
  type: PropTypes.oneOf(["base", "local", "add"]),
  onDownload: PropTypes.func,
  onInfo: PropTypes.func,
  onChat: PropTypes.func,
  onKnowledgeBase: PropTypes.func,
  onDelete: PropTypes.func,
  onClick: PropTypes.func,
  baseModelName: PropTypes.string,
  rebindTargets: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      name: PropTypes.string.isRequired,
    })
  ),
  onRebind: PropTypes.func,
};

export default ModelCard;
