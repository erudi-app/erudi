import React, { useEffect, useState, useRef } from "react";
import { useTranslation } from "react-i18next";
import Sidebar from "../components/Sidebar";
import GradientBox from "../components/GradientBox";
import QuestionInput from "../components/QuestionInput";
import { askArena } from "../services/arenaService.js";
import { canAttachImages, maxImagesForModel } from "../utils/modelCapabilities";
import { defaultsFor } from "../utils/samplingDefaults";
import { Trash, Plus, Square } from "lucide-react";
import HeaderBar from "../components/HeaderBar";
import CustomizePromptModal from "../components/modals/CustomizePromptModal";
import MarkdownRenderer from "../components/MarkdownRenderer";
import apiClient from "../services/api/client";
import { createLogger } from "../utils/logger";

const MAX_PANELS = 4;

// A panel is model-scoped: its sampling seeds from ITS model's resolved
// defaults (#388, `sampling_defaults` on the /llms/local row; the backend
// fallback for a model without hints) and follows the model on switch.
function makePanel(id, modelName = "", models = []) {
  const selectedModel = modelName || (models[0]?.name ?? "");
  return {
    id,
    selectedModel,
    ...defaultsFor(models.find((m) => m.name === selectedModel)),
    customPrompt: "",
    messages: [],
    showPromptModal: false,
    isAnimating: false,
    isRemoving: false,
  };
}

export default function ArenaPage() {
  const log = createLogger("ArenaPage");
  const { t } = useTranslation();

  const [models, setModels] = useState([]);
  const [panels, setPanels] = useState([]);
  const [, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [addButtonAnimating, setAddButtonAnimating] = useState(false);

  const streamingPanels = useRef(new Set());
  const buffersRef = useRef({});
  const flushIntervalRef = useRef(null);
  // One controller per ask: all panel streams share it so Stop aborts the
  // whole comparison at once (#136 H).
  const abortRef = useRef(null);

  useEffect(() => {
    return () => {
      if (flushIntervalRef.current) {
        clearInterval(flushIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    apiClient
      .get("/llms/local")
      .then((data) => {
        setModels(data);
        setPanels([0, 1].map((i) => makePanel(i, data[i % data.length]?.name, data)));
      })
      .catch((err) => log.error("Failed to fetch the local models", err));
  }, []);

  // Switching a panel's model re-defaults its sampling to the new model's
  // values, touched or not (#388): the arena compares MODELS, and a model is
  // compared at its own defaults.
  const handleModelChange = (panelId, newModel) =>
    setPanels((prev) =>
      prev.map((p) =>
        p.id === panelId
          ? {
              ...p,
              selectedModel: newModel,
              ...defaultsFor(models.find((m) => m.name === newModel)),
            }
          : p
      )
    );

  const handleSettingsChange = (panelId, newSettings) =>
    setPanels((prev) => prev.map((p) => (p.id === panelId ? { ...p, ...newSettings } : p)));

  const handleCustomizePrompt = (panelId, show) =>
    setPanels((prev) => prev.map((p) => (p.id === panelId ? { ...p, showPromptModal: show } : p)));

  const handleAsk = async (inputValue, images = []) => {
    if (!inputValue.trim() && images.length === 0) {
      return;
    }
    if (flushIntervalRef.current) {
      clearInterval(flushIntervalRef.current);
    }
    setLoading(true);
    abortRef.current = new AbortController();

    const withPlaceholders = panels.map((panel) => ({
      ...panel,
      messages: [
        ...panel.messages,
        { role: "user", content: inputValue, images },
        { role: "llm", content: "" },
      ],
    }));
    setPanels(withPlaceholders);

    buffersRef.current = {};
    streamingPanels.current = new Set(withPlaceholders.map((p) => p.id));
    withPlaceholders.forEach((p) => {
      buffersRef.current[p.id] = [];
    });

    const flushBuffers = () => {
      const stillStreaming = streamingPanels.current.size > 0;
      const activeIds = stillStreaming
        ? Array.from(streamingPanels.current)
        : Object.keys(buffersRef.current).map(Number);

      const ready = stillStreaming
        ? activeIds.every((id) => buffersRef.current[id].length > 0)
        : activeIds.some((id) => buffersRef.current[id].length > 0);

      if (!ready) {
        return;
      }

      setPanels((prev) =>
        prev.map((p) => {
          const buf = buffersRef.current[p.id];
          if (buf?.length) {
            const nextTok = buf.shift();
            const msgs = [...p.messages];
            msgs[msgs.length - 1] = {
              ...msgs[msgs.length - 1],
              content: msgs[msgs.length - 1].content + nextTok,
            };
            return { ...p, messages: msgs };
          }
          return p;
        })
      );

      if (!stillStreaming && activeIds.every((id) => buffersRef.current[id].length === 0)) {
        clearInterval(flushIntervalRef.current);
        flushIntervalRef.current = null;
        setLoading(false);
      }
    };
    flushIntervalRef.current = setInterval(flushBuffers, 50);

    withPlaceholders.forEach((panel) => {
      const llm = models.find((m) => m.name === panel.selectedModel);
      if (!llm) {
        buffersRef.current[panel.id].push(t("arena:panel.modelNotFound"));
        streamingPanels.current.delete(panel.id);
        if (streamingPanels.current.size === 0) {
          setLoading(false);
        }
        return;
      }

      askArena({
        question: inputValue,
        images,
        llmId: llm.id,
        temperature: panel.temperature,
        topP: panel.topP,
        maxNewTokens: panel.maxTokens,
        customPrompt: panel.customPrompt,
        signal: abortRef.current.signal,
        onStreamChunk: (chunk) => {
          buffersRef.current[panel.id].push(chunk);
        },
      })
        .then(() => {
          streamingPanels.current.delete(panel.id);
          if (streamingPanels.current.size === 0) {
            setLoading(false);
          }
        })
        .catch((err) => {
          // Deliberately swallow AbortError: a user-initiated stop is not an
          // error — panels keep whatever partial text they streamed (#136 H).
          if (err?.name !== "AbortError") {
            log.error(
              `Arena query failed for panel ${panel.id} (model ${panel.selectedModel})`,
              err
            );
            // The error is flagged on the message itself rather than sniffed
            // from its (now language-dependent) text.
            setPanels((prev) =>
              prev.map((p) => {
                if (p.id !== panel.id || p.messages.length === 0) return p;
                const msgs = [...p.messages];
                msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], error: true };
                return { ...p, messages: msgs };
              })
            );
            buffersRef.current[panel.id].push(t("arena:panel.error"));
          }
          streamingPanels.current.delete(panel.id);
          if (streamingPanels.current.size === 0) {
            setLoading(false);
          }
        });
    });

    setInputValue("");
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleAddPanel = () => {
    if (panels.length >= MAX_PANELS) {
      return;
    }

    // Animate the button
    setAddButtonAnimating(true);
    setTimeout(() => setAddButtonAnimating(false), 200);

    const nextId = panels.length ? Math.max(...panels.map((p) => p.id)) + 1 : 0;
    const defaultModel = models[panels.length % models.length]?.name || "";
    const newPanel = {
      ...makePanel(nextId, defaultModel, models),
      isAnimating: true,
    };

    setPanels((prev) => [...prev, newPanel]);

    // Trigger animation
    setTimeout(() => {
      setPanels((prev) => prev.map((p) => (p.id === nextId ? { ...p, isAnimating: false } : p)));
    }, 50);
  };

  const handleDeletePanel = (panelId) => {
    if (panels.length <= 1) {
      return;
    }

    setPanels((prev) => prev.map((p) => (p.id === panelId ? { ...p, isRemoving: true } : p)));

    setTimeout(() => {
      setPanels((prev) => prev.filter((p) => p.id !== panelId));
    }, 300);
  };

  let gridClass = "";
  if (panels.length === 1) {
    gridClass = "grid grid-cols-1 gap-4 w-full h-full";
  } else if (panels.length === 2) {
    gridClass = "grid grid-cols-2 gap-4 w-full";
  } else if (panels.length === 3) {
    gridClass = "grid grid-cols-3 gap-4 w-full";
  } else if (panels.length === 4) {
    gridClass = "grid grid-cols-2 grid-rows-2 gap-4 w-full";
  }

  const renderChatPanel = (panel) => (
    <GradientBox
      key={panel.id}
      className={`flex flex-col mx-2 min-w-[320px] h-[80vh] transition-all duration-300 ease-in-out ${
        panel.isAnimating
          ? "opacity-0 transform scale-95 translate-y-4"
          : panel.isRemoving
            ? "opacity-0 transform scale-95 translate-y-4"
            : "opacity-100 transform scale-100 translate-y-0"
      }`}
    >
      <div className="flex items-start justify-between pb-2 gap-2">
        <div className="flex-1 min-w-0">
          <HeaderBar
            initialTemperature={panel.temperature}
            initialTopP={panel.topP}
            initialMaxTokens={panel.maxTokens}
            maxTokensCap={panel.maxTokensCap}
            // #218: settings take effect at send time. Slider/token edits flow
            // straight into the panel state via onLiveChange, so the displayed
            // value is the value the next run sends - no silent Apply divergence.
            onLiveChange={(s) => handleSettingsChange(panel.id, s)}
            onApply={(s) => handleSettingsChange(panel.id, s)}
            onCustomizePrompt={() => handleCustomizePrompt(panel.id, true)}
            disabled={loading}
            models={models}
            currentModel={panel.selectedModel}
            onModelChange={(m) => handleModelChange(panel.id, m)}
          />
        </div>
        <div className="flex items-center pt-3 flex-shrink-0">
          <button
            className={`p-1 ${
              panels.length <= 1
                ? "text-gray-600 cursor-not-allowed opacity-50"
                : "text-gray-400 hover:text-gray-200 cursor-pointer"
            }`}
            onClick={() => handleDeletePanel(panel.id)}
            disabled={panels.length <= 1}
            title={panels.length <= 1 ? t("arena:panel.deleteLast") : t("arena:panel.delete")}
          >
            <Trash className="w-5 h-5 mt-2" />
          </button>
        </div>
      </div>
      <div
        className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-2 pb-6 custom-scroll"
        style={{
          scrollbarWidth: "none",
          msOverflowStyle: "none",
        }}
      >
        <style>
          {`
            .custom-scroll::-webkit-scrollbar {
              display: none;
            }
          `}
        </style>
        {panel.messages.map((msg, idx) => {
          const isUser = msg.role === "user";
          const isError = msg.error === true;
          return (
            <div
              key={idx}
              className={`rounded-xl px-4 py-2 my-1 max-w-[90%] ${
                isUser
                  ? "bg-emerald-900/40 text-white self-end"
                  : isError
                    ? "text-red-400 self-start"
                    : "text-white self-start"
              }`}
            >
              {isUser || isError ? (
                // Attached images (this session only) render as thumbnails,
                // mirroring the conversation user bubble (#136 C).
                <div className="flex flex-col gap-2">
                  {msg.images?.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {msg.images.map((src, i) => (
                        <img
                          key={i}
                          src={src}
                          alt={t("arena:panel.attachmentAlt", { index: i + 1 })}
                          className="max-h-48 rounded-lg border border-emerald-200/20"
                        />
                      ))}
                    </div>
                  )}
                  {msg.content && (
                    <pre className="whitespace-pre-wrap font-sans">{msg.content}</pre>
                  )}
                </div>
              ) : (
                <MarkdownRenderer content={msg.content} />
              )}
            </div>
          );
        })}
      </div>
    </GradientBox>
  );

  return (
    <div className="flex h-screen">
      <Sidebar disabled={loading} />
      <main
        className="flex-1 flex flex-col bg-[#071b18] relative overflow-auto custom-scroll"
        style={{ paddingBottom: "4rem" }}
      >
        <div className={`flex-1 p-8 ${gridClass}`}>{panels.map(renderChatPanel)}</div>
      </main>
      <div className="fixed bottom-0 left-0 right-0 flex justify-center align-center z-30">
        <div className="max-w-lg w-full mb-8 px-4 py-2 flex items-center gap-3">
          <div className="flex-1">
            {/* Panels can run mixed models: attaching is allowed as soon as ANY
                selected panel model supports vision — the backend safety net
                (#133) strips images per model for the text-only panels. */}
            <QuestionInput
              onSend={handleAsk}
              disabled={loading}
              canAttachImages={panels.some((p) =>
                canAttachImages(models.find((m) => m.name === p.selectedModel))
              )}
              maxImages={Math.min(
                ...panels.map((p) =>
                  maxImagesForModel(models.find((m) => m.name === p.selectedModel))
                )
              )}
            />
          </div>

          {/* Stop button: visible only while a comparison is streaming (#136 H) */}
          {loading && (
            <button
              className="rounded-full -mb-1"
              onClick={handleStop}
              title={t("arena:controls.stop")}
              aria-label={t("arena:controls.stop")}
            >
              <div
                className={[
                  "relative flex items-center justify-center w-10 h-10 rounded-full overflow-hidden",
                  "border border-emerald-400/20",
                  "bg-emerald-900/30 backdrop-blur-[10px] saturate-[1.3]",
                  "shadow-[0_10px_30px_-6px_rgba(0,0,0,0.5),0_2px_6px_-1px_rgba(0,0,0,0.45)]",
                ].join(" ")}
              >
                <Square
                  className="w-4 h-4 text-white relative z-10"
                  fill="currentColor"
                  strokeWidth={2}
                />
              </div>
            </button>
          )}

          {/* Add Panel Button */}
          <button
            className={`rounded-full transition-all duration-200 -mb-1 ${
              panels.length >= MAX_PANELS ? "opacity-50 cursor-not-allowed" : ""
            } ${addButtonAnimating ? "transform scale-110" : "transform scale-100"}`}
            onClick={handleAddPanel}
            disabled={panels.length >= MAX_PANELS}
            title={
              panels.length >= MAX_PANELS
                ? t("arena:controls.maxPanels", { max: MAX_PANELS })
                : t("arena:controls.addPanel")
            }
          >
            {/* Glassy effect container */}
            <div
              className={[
                "relative flex items-center justify-center w-10 h-10 rounded-full overflow-hidden",
                "border border-emerald-400/20",
                "bg-emerald-900/30 backdrop-blur-[10px] saturate-[1.3]",
                "shadow-[0_10px_30px_-6px_rgba(0,0,0,0.5),0_2px_6px_-1px_rgba(0,0,0,0.45)]",
              ].join(" ")}
            >
              {/* Frost overlays with emerald tint */}
              <div
                aria-hidden
                className="pointer-events-none absolute inset-0 rounded-full mix-blend-overlay"
                style={{
                  background:
                    "linear-gradient(180deg, rgba(16,185,129,0.12) 0%, rgba(16,185,129,0.06) 28%, rgba(16,185,129,0.02) 60%, rgba(16,185,129,0) 100%)",
                }}
              />
              <div
                aria-hidden
                className="pointer-events-none absolute inset-0 rounded-full"
                style={{
                  boxShadow:
                    "inset 0 1px 0 rgba(16,185,129,0.15), inset 0 -1px 0 rgba(16,185,129,0.08)",
                }}
              />

              {/* Plus icon from Lucide */}
              <Plus className="w-6 h-6 text-white relative z-10" strokeWidth={2} />
            </div>
          </button>
        </div>
      </div>

      {/* Render all modals at top level to ensure proper z-index stacking */}
      {panels.map(
        (panel) =>
          panel.showPromptModal && (
            <CustomizePromptModal
              key={`modal-${panel.id}`}
              isOpen={panel.showPromptModal}
              onClose={() => handleCustomizePrompt(panel.id, false)}
              customPrompt={panel.customPrompt}
              onSave={(newPrompt) =>
                handleSettingsChange(panel.id, {
                  customPrompt: newPrompt,
                })
              }
            />
          )
      )}
    </div>
  );
}
