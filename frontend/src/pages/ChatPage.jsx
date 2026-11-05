import React, { useEffect, useState, useCallback } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Sidebar from "../components/Sidebar";
import ChatCollapsibleSection from "../components/ChatCollapsibleSection";
import GradientBox from "../components/GradientBox";
import QuestionInput from "../components/QuestionInput";
import ConversationHeader from "../components/ConversationHeader";
import CommandPopups from "../components/CommandPopups";
import ErrorModal from "../components/modals/ErrorModal";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown } from "lucide-react";
import logoErudi from "../assets/images/logos/logoerudifinal.png";
import { API_BASE_URL } from "../config/api.js";
import apiClient, { tracedFetch } from "../services/api/client";
import { useDownloadModal } from "../contexts/DownloadModalContext";
import { createLogger } from "../utils/logger";
import { conversationPath } from "../utils/routes";
import { canAttachImages } from "../utils/modelCapabilities";
import { hasMissingWeights } from "../utils/modelWeights";
import { COMMAND_SCOPES, commandsFor, commandsOutside } from "../utils/slashCommands";
import useConversationSettings, { DRAFT } from "../shared/hooks/useConversationSettings";
import CommandNotice from "../components/CommandNotice";

const log = createLogger("ChatPage");

// The slash commands the new-conversation page offers: they edit the draft
// the conversation is created from. /compact and /context need a
// conversation: typed here, they send nothing and a notice says why.
const COMMANDS = commandsFor(COMMAND_SCOPES.DRAFT);
const NEEDS_CONVERSATION = commandsOutside(COMMAND_SCOPES.DRAFT);

export default function ChatPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const location = useLocation();
  const { completionCount } = useDownloadModal();

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [conversations, setConversations] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [collapsed, setCollapsed] = useState(false);
  const [isLanguageWarningExpanded, setIsLanguageWarningExpanded] = useState(false);

  // The draft the conversation is created from (#609): the picked model's
  // resolved defaults, and what the user set through /config, /effort and
  // the header panel. Only what the user set is sent at creation; the
  // backend resolves the rest for the model.
  const draft = useConversationSettings(DRAFT);
  const { selectModel, ready, createPayload } = draft;
  // The command being run (its popup is open), or null.
  const [openCommand, setOpenCommand] = useState(null);
  // Why a command typed here did nothing, or null; first, the notice a
  // conversation page handed over when it sent the user here (a /compact
  // that found its conversation deleted).
  const [commandNotice, setCommandNotice] = useState(() => location.state?.notice ?? null);
  // Read once: the history entry forgets it, so a reload of the window
  // (which keeps the entry's state) never shows it again.
  const handedOverNotice = Boolean(location.state?.notice);
  useEffect(() => {
    if (handedOverNotice) {
      navigate(`${location.pathname}${location.search}`, { replace: true, state: null });
    }
  }, [handedOverNotice, location.pathname, location.search, navigate]);
  const runCommand = (command) => {
    setCommandNotice(null);
    setOpenCommand(command);
  };
  const explainUnavailable = (command) =>
    setCommandNotice({
      tone: "info",
      text: t("chat:commands.needsConversation", { command: `/${command.name}` }),
    });

  const toggleSidebar = () => {
    setCollapsed((prev) => !prev);
  };

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

  const llm = models.find((m) => m.name === selectedModel);

  // The draft follows the picked model: its defaults are read, and a switch
  // brings the new model's values (the /model rule).
  useEffect(() => {
    if (llm) selectModel(llm);
  }, [llm, selectModel]);

  const handleConversationClick = (id) => {
    navigate(conversationPath(id));
  };

  const handleAsk = useCallback(
    async (question, images = [], imagePaths = [], attachmentPaths = []) => {
      if (!llm || hasMissingWeights(llm)) {
        log.error("Selected model not found or has no weights on disk");
        return;
      }
      try {
        // 1. Create the conversation from the draft: the model and the values
        // the user set; the backend resolves every other one for the model.
        // A model picked just before sending: its defaults first decide what
        // the user set that still holds.
        await ready();
        const res = await tracedFetch(`${API_BASE_URL}/conversations/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(createPayload()),
        });
        if (!res.ok) {
          throw new Error(t("chat:errors.createConversation"));
        }
        const conversation = await res.json();

        // 2. Hand the question over: the conversation page reads the created
        // conversation and sends the first turn on its values.
        navigate(conversationPath(conversation.id), {
          state: {
            initialQuestion: question,
            initialImages: images,
            initialImagePaths: imagePaths,
            initialAttachments: attachmentPaths,
          },
        });
      } catch (err) {
        log.error("Failed to start conversation:", err);
        setErrorMessage(
          t("chat:errors.startConversation", { error: err.message || t("chat:errors.network") })
        );
      }
    },
    [llm, navigate, ready, createPayload, t]
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
          <div className="w-[700px] max-w-full">
            {/* Logo Erudi */}
            <div className="flex justify-center mb-8">
              <img src={logoErudi} alt="Erudi" className="h-20 w-auto" />
            </div>

            {/* The same header as a conversation; its commands edit the draft. */}
            <ConversationHeader
              modelName={selectedModel}
              attention={!selectedModel}
              commands={COMMANDS}
              onCommand={runCommand}
            />

            {/* Question Input - Always visible */}
            <div className="mt-6">
              <CommandNotice notice={commandNotice} onDismiss={() => setCommandNotice(null)} />
              <QuestionInput
                onSend={handleAsk}
                disabled={!selectedModel}
                placeholder={t("chat:composer.placeholderNew")}
                canAttachImages={canAttachImages(llm)}
                commands={COMMANDS}
                onCommand={runCommand}
                unavailableCommands={NEEDS_CONVERSATION}
                onUnavailableCommand={explainUnavailable}
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
        )}
      </main>

      {/* Error Popup */}
      <ErrorModal errorMessage={errorMessage} onClose={() => setErrorMessage("")} />

      {/* The popup of the command being run (/config, /effort, /model). */}
      <CommandPopups
        command={openCommand}
        onClose={() => setOpenCommand(null)}
        settings={draft}
        model={llm}
        models={models}
        onPickModel={(model) => setSelectedModel(model.name)}
      />
    </div>
  );
}
