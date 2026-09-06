import React, { useEffect, useState, useCallback, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Sidebar from "../components/Sidebar";
import ChatCollapsibleSection from "../components/ChatCollapsibleSection";
import GradientBox from "../components/GradientBox";
import QuestionInput from "../components/QuestionInput";
import CustomizePromptModal from "../components/modals/CustomizePromptModal";
import Tooltip from "../components/Tooltip";
import ToggleSwitch from "../components/ToggleSwitch";
import ErrorModal from "../components/modals/ErrorModal";
import { motion, AnimatePresence } from "framer-motion";
import { SlidersHorizontal, ChevronDown, HelpCircle } from "lucide-react";
import logoErudi from "../assets/images/logos/logoerudifinal.png";
import { API_BASE_URL } from "../config/api.js";
import apiClient, { tracedFetch } from "../services/api/client";
import { useDownloadModal } from "../contexts/DownloadModalContext";
import { createLogger } from "../utils/logger";
import { conversationPath } from "../utils/routes";
import { canAttachImages } from "../utils/modelCapabilities";
import { hasMissingWeights } from "../utils/modelWeights";
import { defaultsFor, hasNoPublisherRecommendation } from "../utils/samplingDefaults";
import { formatNumber } from "../i18n/format";

const log = createLogger("ChatPage");

// Temperature / top-p read-outs: two decimals in the active locale.
const TWO_DECIMALS = { minimumFractionDigits: 2, maximumFractionDigits: 2 };

export default function ChatPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { completionCount } = useDownloadModal();

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [conversations, setConversations] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [collapsed, setCollapsed] = useState(false);
  const [isLanguageWarningExpanded, setIsLanguageWarningExpanded] = useState(false);

  // Parameters state (#388): seeded from the selected model's resolved
  // defaults (`sampling_defaults` on the /llms/local row; the backend
  // fallback 0.2 / 0.95 / 1024 before a model is known or for a model
  // without hints). Every model switch re-defaults them to the new model's
  // values, touched or not (maintainer decision 1); a touched value persists
  // while the model stays the same and is what the creation POST sends.
  const [settings, setSettings] = useState(() => defaultsFor(null));
  const patchSettings = (patch) => setSettings((prev) => ({ ...prev, ...patch }));
  const [customPrompt, setCustomPrompt] = useState("");
  const [showPromptModal, setShowPromptModal] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  // Web-search toggle (#310): null until the GLOBAL default is fetched, so an
  // untouched panel inherits it server-side (the creation payload omits the
  // field while null). Once fetched or flipped, the explicit value is sent
  // and the FIRST turn already honors it.
  const [webSearch, setWebSearch] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await apiClient.get("/user_settings/");
        if (!cancelled) setWebSearch(Boolean(data?.web_search_enabled));
      } catch (error) {
        // The backend applies its own default: degraded, not failed.
        log.warn("Failed to fetch the global web search default", error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Refs for dropdown
  const dropdownRef = useRef(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const toggleSidebar = () => {
    setCollapsed((prev) => !prev);
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const fetchModels = useCallback(() => {
    apiClient
      .get("/llms/local")
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setModels(data);
          // Keep the user's current selection on refreshes while it is still
          // selectable; otherwise default to the first model whose weights
          // are on disk. An orphaned KB assistant (weights_available === false,
          // #376) is listed but never auto-selected: a turn on it can only
          // fail. With nothing selectable the selection stays empty and the
          // composer disabled.
          setSelectedModel((prev) => {
            const current = data.find((m) => m.name === prev);
            if (current && !hasMissingWeights(current)) return prev;
            return data.find((m) => !hasMissingWeights(m))?.name ?? "";
          });
        }
      })
      .catch((err) => {
        log.error("Failed to fetch the local models", err);
        setErrorMessage(
          t("chat:errors.loadModels", { error: err.message || t("chat:errors.network") })
        );
      });
  }, [t]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // A completed download bumps the context's completionCount, whichever entry
  // point started it. Refresh the model list on every tick so a user sitting
  // on Chat during a download gets the composer without navigating away and
  // back (#303) — same pattern as LandingPage (#205).
  useEffect(() => {
    if (!completionCount) {
      return;
    }
    fetchModels();
  }, [completionCount, fetchModels]);

  useEffect(() => {
    const fetchConversations = async () => {
      try {
        const data = await apiClient.get("/conversations/");
        const sorted = data.sort(
          (a, b) => new Date(b.last_message_time) - new Date(a.last_message_time)
        );
        setConversations(sorted);
      } catch (err) {
        log.error("Failed to fetch conversations:", err);
        setErrorMessage(
          t("chat:errors.loadConversations", { error: err.message || t("chat:errors.network") })
        );
      }
    };

    fetchConversations();
  }, [t]);

  // Preselect the model referenced by the ?model= URL parameter (#223).
  // Producers pass either the numeric id or the exact name; the id from the
  // backend is a number while the URL value is a string, so compare through
  // String(). Match the id first, then the decoded name. An unknown reference
  // keeps the default selection (first local model).
  useEffect(() => {
    const modelParam = searchParams.get("model");
    if (!modelParam || models.length === 0) return;

    const foundModel =
      models.find((model) => String(model.id) === modelParam) ??
      models.find((model) => model.name === modelParam);

    if (foundModel && hasMissingWeights(foundModel)) {
      // A deep link to an orphaned assistant (#376) keeps the default pick.
      log.warn("Model from URL parameter has no weights on disk:", modelParam);
    } else if (foundModel) {
      log.log("Setting model from URL parameter:", foundModel);
      setSelectedModel(foundModel.name);
    } else {
      log.warn("Model not found for URL parameter:", modelParam);
    }
  }, [searchParams, models]); // Re-run when the models list arrives (#211 lesson)

  // Re-seed the panel from the selected model on every switch (#388): the
  // previous values were the previous model's.
  useEffect(() => {
    const model = models.find((m) => m.name === selectedModel);
    if (!model) return;
    setSettings(defaultsFor(model));
  }, [selectedModel, models]);

  const handleConversationClick = (id) => {
    navigate(conversationPath(id));
  };

  const handleAsk = useCallback(
    async (question, images = [], imagePaths = []) => {
      const llm = models.find((m) => m.name === selectedModel);
      if (!llm || hasMissingWeights(llm)) {
        log.error("Selected model not found or has no weights on disk");
        return;
      }
      try {
        // 1. Create a new conversation with the specified parameters
        const res = await tracedFetch(`${API_BASE_URL}/conversations/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            llm_id: llm.id,
            temperature: settings.temperature,
            top_p: settings.topP,
            max_tokens: settings.maxTokens,
            custom_prompt: customPrompt,
            // null (global default not fetched yet) -> omit: the backend
            // copies the global setting at creation (#310).
            ...(webSearch === null ? {} : { web_search_enabled: webSearch }),
          }),
        });
        if (!res.ok) {
          throw new Error(t("chat:errors.createConversation"));
        }
        const conversation = await res.json();

        // 2. Redirect to ConversationPage and pass the question AND parameters
        navigate(conversationPath(conversation.id), {
          state: {
            initialQuestion: question,
            initialImages: images,
            initialImagePaths: imagePaths,
            initialSettings: settings,
            initialCustomPrompt: customPrompt,
          },
        });
      } catch (err) {
        log.error("Failed to start conversation:", err);
        setErrorMessage(
          t("chat:errors.startConversation", { error: err.message || t("chat:errors.network") })
        );
      }
    },
    [models, selectedModel, navigate, settings, customPrompt, webSearch, t]
  );

  const handleRename = (id, newName) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, name: newName } : c)));
  };

  const handleDelete = (id) => {
    setConversations((prev) => prev.filter((conv) => conv.id !== id));
  };

  const handleRefreshConversations = async () => {
    try {
      const data = await apiClient.get("/conversations/");
      const sorted = data.sort(
        (a, b) => new Date(b.last_message_time) - new Date(a.last_message_time)
      );
      setConversations(sorted);
    } catch (err) {
      log.error("Failed to refresh conversations:", err);
      setErrorMessage(
        t("chat:errors.refreshConversations", { error: err.message || t("chat:errors.network") })
      );
    }
  };

  // Utility functions for HeaderBar-like styling
  const TooltipIcon = ({ id, side = "right" }) => {
    const text =
      id === "temperature"
        ? t("chat:header.tooltips.temperature")
        : id === "top-p"
          ? t("chat:header.tooltips.topP")
          : id === "prompt"
            ? t("chat:header.tooltips.prompt")
            : id === "web-search"
              ? t("chat:header.tooltips.webSearch")
              : "";
    return (
      <Tooltip content={text} side={side} width="w-64">
        <HelpCircle className="w-4 h-4 text-gray-400 hover:text-emerald-400 transition-colors cursor-help" />
      </Tooltip>
    );
  };

  const sliderBg = (value, max = 1) => {
    const pct = Math.round((value / max) * 100);
    return {
      background: `linear-gradient(to right, #25C08A 0%, #1EAB78 ${pct}%, rgba(255,255,255,0.06) ${pct}%, rgba(255,255,255,0.06) 100%)`,
    };
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar showCollapsible={true} onToggleSidebar={toggleSidebar} collapsed={collapsed} />

      {/* barre latérale */}
      <aside
        className={`${
          collapsed ? "w-0 opacity-0" : "w-80 opacity-100 p-6 space-y-6"
        } relative bg-[#272727] text-white transition-all duration-300`}
      >
        {/* Content only when expanded */}
        {!collapsed && (
          <>
            <h1 className="text-3xl font-bold">{t("chat:history.title")}</h1>

            {/*<ChatCollapsibleSection title="Hot Chats"
              disabled={loading}
            />} coming in next version*/}
            <ChatCollapsibleSection
              title={t("chat:history.previousChats")}
              items={conversations}
              onItemClick={handleConversationClick}
              onRename={handleRename}
              onDelete={handleDelete}
              onRefresh={handleRefreshConversations}
            />
          </>
        )}
      </aside>

      {/* zone centrale */}
      <main className="flex-1 bg-[#071b18] flex items-center justify-center relative overflow-hidden py-12 px-8">
        {/* Si aucun modèle local */}
        {models.length === 0 ? (
          <GradientBox className="w-[700px] max-w-full">
            <div className="text-white text-center py-10">{t("chat:empty.noModels")}</div>
            {/* Language Warning */}
            <div className="flex justify-center px-2 pb-1">
              <div className="w-[700px] max-w-full">
                <div
                  className={[
                    "relative w-full rounded-[26px] overflow-hidden",
                    "bg-[rgba(64,35,22,0.45)] backdrop-blur-[18px] saturate-[1.4]",
                    "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,200,150,0.06)]",
                  ].join(" ")}
                >
                  <div
                    aria-hidden
                    className="absolute inset-0 pointer-events-none rounded-[26px] mix-blend-overlay"
                    style={{
                      background:
                        "linear-gradient(to bottom, rgba(255,180,100,0.18), rgba(255,180,100,0) 40%)",
                    }}
                  />

                  {/* Content */}
                  <div className="relative z-10 p-6">
                    <h2 className="text-lg font-semibold tracking-tight text-orange-100 mb-3">
                      {t("chat:languageNote.title")}
                    </h2>
                    <p className="text-sm text-orange-200/80 mb-3">{t("chat:languageNote.body")}</p>
                    <p className="text-sm text-orange-200/70 italic">
                      {t("chat:languageNote.frenchAside")}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </GradientBox>
        ) : (
          /* Interface de création de chat avec design HeaderBar */
          <div className="w-[700px] max-w-full">
            {/* Logo Erudi */}
            <div className="flex justify-center mb-8">
              <img src={logoErudi} alt="Erudi" className="h-20 w-auto" />
            </div>

            {/* HeaderBar-like container */}
            <div
              className={[
                "hb-scope relative w-full rounded-[26px] mb-6",
                "border border-white/10",
                "bg-[rgba(22,40,36,0.45)] backdrop-blur-[18px] saturate-[1.4]",
                "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)]",
              ].join(" ")}
            >
              {/* Glossy overlays */}
              <div
                aria-hidden
                className="absolute inset-0 pointer-events-none rounded-[26px] mix-blend-overlay"
                style={{
                  background:
                    "linear-gradient(to bottom, rgba(255,255,255,0.18), rgba(255,255,255,0) 40%)",
                }}
              />

              <div className="relative z-10 p-5">
                <style>{`
                  .hb-scope input.hb-range { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 999px; outline: none; }
                  .hb-scope input.hb-range::-webkit-slider-thumb {
                    -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; border: 0; cursor: pointer;
                    background: radial-gradient(circle at 30% 30%, #ffffff, #d9e4dd 60%, #b7c6c0 100%);
                    box-shadow: 0 2px 6px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.4), inset 0 1px 2px rgba(255,255,255,0.7);
                    transition: transform .25s ease, box-shadow .25s ease;
                  }
                  .hb-scope input.hb-range:hover::-webkit-slider-thumb { transform: scale(1.07); }
                  .hb-scope input.hb-range:active::-webkit-slider-thumb { transform: scale(.9); }
                  .hb-scope input.hb-range:focus-visible::-webkit-slider-thumb {
                    box-shadow: 0 0 0 4px rgba(37,192,138,0.35), 0 2px 6px rgba(0,0,0,0.55), inset 0 1px 2px rgba(255,255,255,0.8);
                  }
                  .hb-scope input.hb-range::-moz-range-track { height: 6px; background: rgba(255,255,255,0.06); border-radius: 999px; }
                  .hb-scope input.hb-range::-moz-range-thumb {
                    width: 18px; height: 18px; border-radius: 50%; border: 0; cursor: pointer;
                    background: radial-gradient(circle at 30% 30%, #ffffff, #d9e4dd 60%, #b7c6c0 100%);
                    box-shadow: 0 2px 6px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.4), inset 0 1px 2px rgba(255,255,255,0.7);
                  }
                  .hb-scope input.hb-range:focus-visible::-moz-range-thumb {
                    box-shadow: 0 0 0 4px rgba(37,192,138,0.35), 0 2px 6px rgba(0,0,0,0.55), inset 0 1px 2px rgba(255,255,255,0.8);
                  }
                `}</style>

                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3 flex-wrap min-w-0">
                    <h3 className="text-[1.15rem] font-semibold tracking-tight text-[#F2F7F4] truncate">
                      {t("chat:header.chatWith")}
                    </h3>

                    <div
                      ref={dropdownRef}
                      className={[
                        "inline-flex items-center rounded-lg relative z-50",
                        "px-3.5 py-1.5 text-sm",
                        "border transition",
                        "bg-white/5 hover:bg-white/10 border-white/10 hover:border-white/20",
                        "backdrop-blur-sm text-gray-100",
                        "max-w-[100%] cursor-pointer",
                      ].join(" ")}
                      onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                    >
                      <div
                        className="font-medium truncate pr-5 max-w-[150px]"
                        title={selectedModel}
                      >
                        {selectedModel || t("chat:header.selectModelPlaceholder")}
                      </div>
                      <ChevronDown
                        size={16}
                        className={`opacity-70 shrink-0 transition-transform absolute right-3 ${
                          isDropdownOpen ? "rotate-180" : ""
                        }`}
                      />

                      {/* Custom Dropdown */}
                      {isDropdownOpen && (
                        <div className="absolute top-full left-0 right-0 mt-1 bg-[#2a2a2a] border border-white/20 rounded-lg shadow-lg z-[9999] max-h-60 overflow-y-auto">
                          {models.map((m) => {
                            // Orphaned assistant (#376): listed but not
                            // selectable, with the remedy spelled out.
                            const weightsMissing = hasMissingWeights(m);
                            return (
                              <div
                                key={m.id ?? m.name}
                                data-testid={`model-option-${m.id ?? m.name}`}
                                aria-disabled={weightsMissing}
                                className={[
                                  "px-3 py-2 border-b border-white/10 last:border-b-0",
                                  weightsMissing
                                    ? "text-gray-500 cursor-not-allowed"
                                    : "hover:bg-white/10 cursor-pointer text-gray-100",
                                ].join(" ")}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  if (weightsMissing) return;
                                  setSelectedModel(m.name);
                                  setIsDropdownOpen(false);
                                }}
                              >
                                {m.name}
                                {weightsMissing && (
                                  <div className="text-[11px] leading-snug text-gray-500">
                                    {t("chat:header.weightsMissingHint")}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>

                  <button
                    type="button"
                    aria-label={t("chat:header.toggleSettings")}
                    onClick={() => setIsSettingsOpen((v) => !v)}
                    className={[
                      "inline-flex items-center justify-center",
                      "w-9 h-9 rounded-xl",
                      "bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20",
                      "text-gray-300 hover:text-emerald-400 transition",
                      "shrink-0",
                    ].join(" ")}
                  >
                    <SlidersHorizontal size={18} />
                  </button>
                </div>

                <AnimatePresence initial={false}>
                  {isSettingsOpen && (
                    <motion.div
                      key="controls"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ type: "tween", duration: 0.3 }}
                      className="overflow-hidden"
                    >
                      <div className="mt-6 grid gap-6 md:grid-cols-2">
                        <div className="flex flex-col gap-6">
                          <div className="relative">
                            <div className="flex items-center gap-1.5 mb-1">
                              <span className="text-[0.72rem] uppercase tracking-wide font-semibold text-gray-300/80">
                                {t("chat:header.creativity")}
                              </span>
                              <TooltipIcon id="temperature" side="right" />
                              <span className="ml-auto text-[11px] font-semibold text-emerald-200/90 bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-400/25">
                                {formatNumber(settings.temperature, TWO_DECIMALS)}
                              </span>
                            </div>

                            <div className="relative pt-1">
                              <input
                                type="range"
                                min="0"
                                max="2"
                                step="0.01"
                                value={settings.temperature}
                                onChange={(e) =>
                                  patchSettings({ temperature: parseFloat(e.target.value) })
                                }
                                className="hb-range w-full rounded-full bg-white/5 cursor-pointer"
                                style={sliderBg(settings.temperature, 2)}
                              />
                            </div>
                          </div>

                          <div className="relative">
                            <div className="flex items-center gap-1.5 mb-1">
                              <span className="text-[0.72rem] uppercase tracking-wide font-semibold text-gray-300/80">
                                {t("chat:header.diversity")}
                              </span>
                              <TooltipIcon id="top-p" side="right" />
                              <span className="ml-auto text-[11px] font-semibold text-emerald-200/90 bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-400/25">
                                {formatNumber(settings.topP, TWO_DECIMALS)}
                              </span>
                            </div>

                            <div className="relative pt-1">
                              <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.01"
                                value={settings.topP}
                                onChange={(e) =>
                                  patchSettings({ topP: parseFloat(e.target.value) })
                                }
                                className="hb-range w-full rounded-full bg-white/5 cursor-pointer"
                                style={sliderBg(settings.topP)}
                              />
                            </div>
                          </div>
                          {hasNoPublisherRecommendation(
                            models.find((m) => m.name === selectedModel)
                          ) && (
                            <p
                              data-testid="no-publisher-recommendation"
                              className="-mt-3 text-[11px] leading-snug text-gray-400/80"
                            >
                              {t("chat:header.noPublisherRecommendation")}
                            </p>
                          )}
                        </div>

                        <div className="flex flex-col justify-center gap-6">
                          <div>
                            <div className="grid grid-cols-2 items-start justify-items-start gap-x-6 gap-y-2 mb-2">
                              <div>
                                <span className="text-[0.72rem] uppercase tracking-wide font-semibold text-gray-300/80">
                                  {t("chat:header.maxTokens")}
                                </span>
                              </div>
                              {/* Controls row */}
                              <div className="inline-flex items-center rounded-md bg-white/10 border border-white/20 shadow p-0 m-0 w-fit justify-self-start">
                                <input
                                  type="number"
                                  min="1"
                                  max={settings.maxTokensCap}
                                  value={settings.maxTokens}
                                  onChange={(e) =>
                                    patchSettings({
                                      maxTokens: parseInt(e.target.value || "0", 10),
                                    })
                                  }
                                  className="bg-transparent border-0 outline-none w-28 text-sm font-semibold text-gray-100 text-center appearance-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                                />
                              </div>
                            </div>
                            <div className="mt-4 flex items-center gap-3">
                              <span className="text-[0.72rem] uppercase tracking-wide font-semibold text-gray-300/80">
                                {t("chat:header.webSearch")}
                              </span>
                              <TooltipIcon id="web-search" side="right" />
                              <div className="ml-auto">
                                <ToggleSwitch
                                  checked={webSearch === true}
                                  onChange={(next) => setWebSearch(next)}
                                  label={t("chat:header.webSearchToggle")}
                                />
                              </div>
                            </div>
                          </div>

                          <div className="flex flex-row items-center gap-3 w-full">
                            <div className="flex items-center gap-2 w-full">
                              <button
                                type="button"
                                onClick={() => setShowPromptModal(true)}
                                className={[
                                  "rounded-md font-semibold",
                                  "px-5 py-2 text-[0.9rem]",
                                  "bg-emerald-800 hover:bg-emerald-900 text-white",
                                  "border border-white/20 shadow",
                                  "transition active:scale-95",
                                ].join(" ")}
                              >
                                {t("chat:header.customizePrompt")}
                              </button>
                              <div>
                                <TooltipIcon id="prompt" side="top-left" />
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Question Input - Always visible */}
                <div className="mt-6">
                  <QuestionInput
                    onSend={handleAsk}
                    disabled={!selectedModel}
                    placeholder={t("chat:composer.placeholderNew")}
                    canAttachImages={canAttachImages(models.find((m) => m.name === selectedModel))}
                  />
                </div>
                {/* Language Warning */}
                <div className="flex justify-center mt-6 px-2 pb-1">
                  <div className="w-[700px] max-w-full">
                    <div
                      className={[
                        "relative w-full rounded-[26px] overflow-hidden cursor-pointer transition-all duration-300",
                        "bg-[rgba(64,35,22,0.45)] backdrop-blur-[18px] saturate-[1.4]",
                        "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,200,150,0.06)]",
                        "hover:border-orange-600/40",
                      ].join(" ")}
                      onClick={() => setIsLanguageWarningExpanded(!isLanguageWarningExpanded)}
                    >
                      <div
                        aria-hidden
                        className="absolute inset-0 pointer-events-none rounded-[26px] mix-blend-overlay"
                        style={{
                          background:
                            "linear-gradient(to bottom, rgba(255,180,100,0.18), rgba(255,180,100,0) 40%)",
                        }}
                      />

                      {/* Content */}
                      <div className="relative z-10 p-4 px-6">
                        <div className="flex items-center justify-between">
                          <h2 className="text-base font-semibold tracking-tight text-orange-100">
                            {t("chat:languageNote.title")}
                          </h2>
                          <motion.div
                            animate={{ rotate: isLanguageWarningExpanded ? 180 : 0 }}
                            transition={{ duration: 0.3 }}
                          >
                            <ChevronDown className="w-5 h-5 text-orange-100" />
                          </motion.div>
                        </div>

                        <AnimatePresence>
                          {isLanguageWarningExpanded && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: "auto", opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.3 }}
                              className="overflow-hidden"
                            >
                              <p className="text-sm text-orange-200/80 mb-3 mt-3">
                                {t("chat:languageNote.body")}
                              </p>
                              <p className="text-sm text-orange-200/70 italic">
                                {t("chat:languageNote.frenchAside")}
                              </p>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Error Popup */}
      <ErrorModal errorMessage={errorMessage} onClose={() => setErrorMessage("")} />

      {/* Customize Prompt Modal */}
      <CustomizePromptModal
        isOpen={showPromptModal}
        onClose={() => setShowPromptModal(false)}
        customPrompt={customPrompt}
        onSave={(newPrompt) => setCustomPrompt(newPrompt)}
        title={t("chat:prompt.title")}
      />
    </div>
  );
}
