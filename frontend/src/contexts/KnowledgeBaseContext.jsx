// src/contexts/KnowledgeBaseContext.jsx
import React, { createContext, useContext, useState, useCallback, useRef } from "react";
import ReactDOM from "react-dom";
import { Trans, useTranslation } from "react-i18next";
import SpinnerDots from "../components/Spinner";
import { API_BASE_URL } from "../config/api";
import { tracedFetch } from "../services/api/client";
import { createLogger } from "../utils/logger";
const log = createLogger("KnowledgeBaseContext");

const KnowledgeBaseContext = createContext();

export function KnowledgeBaseProvider({ children }) {
  const { t } = useTranslation();
  const [task, setTask] = useState(null);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isStarting, setIsStarting] = useState(false); // For the initial API call
  const [showSpinner, setShowSpinner] = useState(false); // For bottom-left spinner
  const [status, setStatus] = useState("idle");
  const [, setErrorMessage] = useState("");
  const [, setAssistantId] = useState(null);

  const intervalRef = useRef(null);
  const callbacksRef = useRef({ onComplete: null, onError: null });

  const open = useCallback((knowledgeBaseTask, { onComplete, onError } = {}) => {
    log.log("KnowledgeBase context open function called with:", knowledgeBaseTask);
    setTask(knowledgeBaseTask);
    callbacksRef.current = { onComplete, onError };
    setErrorMessage("");
    setIsConfirmOpen(true);
    log.log("setIsConfirmOpen set to true");
  }, []);

  const cancelConfirm = useCallback(() => setIsConfirmOpen(false), []);

  const startCreation = useCallback(async () => {
    setIsConfirmOpen(false);
    setIsStarting(true); // Show spinner in button place
    setErrorMessage("");

    try {
      // Start the knowledge base creation API call
      const response = await tracedFetch(`${API_BASE_URL}/knowledge_base/create`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          paths: task.paths,
          selectedModel: task.selectedModel,
          modelName: task.modelName,
          description: task.description,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          t("knowledgeBase:creation.errors.startFailed", {
            status: response.status,
            detail: errorData.detail || t("knowledgeBase:creation.errors.unknownDetail"),
          })
        );
      }

      const result = await response.json();
      const newAssistantId = result.model_id;

      // Update the assistantId for status polling
      setAssistantId(newAssistantId);

      // Switch from button spinner to bottom-left spinner
      setIsStarting(false);
      setIsCreating(true);
      setShowSpinner(true);
      setStatus("pending");

      intervalRef.current = setInterval(() => {
        checkCreationStatus(newAssistantId);
      }, 2000);
    } catch (err) {
      log.error("Knowledge base creation error:", err);
      setIsStarting(false);
      setErrorMessage(err.message || t("knowledgeBase:creation.errors.startFailedGeneric"));
      callbacksRef.current.onError?.(err.message);
    }
  }, [task, t]);

  const checkCreationStatus = useCallback(
    async (assistantId) => {
      try {
        const res = await tracedFetch(`${API_BASE_URL}/knowledge_base/${assistantId}/status`);
        if (!res.ok) {
          throw new Error(`Server responded with ${res.status}: ${res.statusText}`);
        }
        const data = await res.json();

        setStatus(data.status);

        if (data.status === "completed" || data.status === "failed") {
          clearInterval(intervalRef.current);
          setIsCreating(false);
          setShowSpinner(false);
          if (data.status === "completed") {
            callbacksRef.current.onComplete?.();
          } else {
            // The backend logged the traceback under this job; this side
            // records that the assistant the user asked for was not built.
            log.error(`Knowledge base job failed for assistant ${assistantId}`, {
              error: data.error_message || null,
            });
            const errorMsg =
              data.error_message || t("knowledgeBase:creation.errors.failedUnexpectedly");
            setErrorMessage(errorMsg);
            callbacksRef.current.onError?.(errorMsg);
          }
        }
      } catch (err) {
        log.error(`Status check failed for assistant ${assistantId}; polling stopped`, err);
        clearInterval(intervalRef.current);
        setIsCreating(false);
        setShowSpinner(false);
        const errorMsg = t("knowledgeBase:creation.errors.statusCheckFailed");
        setErrorMessage(errorMsg);
        callbacksRef.current.onError?.(errorMsg);
      }
    },
    [t]
  );

  const closeModal = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    setIsCreating(false);
    setIsStarting(false);
    setShowSpinner(false);
    setIsConfirmOpen(false);
    setTask(null);
    setStatus("idle");
    setErrorMessage("");
    setAssistantId(null);
  }, []);

  return (
    <KnowledgeBaseContext.Provider
      value={{
        open,
        isCreating,
        isStarting,
        closeModal,
      }}
    >
      {children}

      {/* Confirmation Modal */}
      {isConfirmOpen &&
        task &&
        ReactDOM.createPortal(
          <div className="fixed inset-0 flex items-center justify-center z-50">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black bg-opacity-70" onClick={cancelConfirm} />

            {/* Modal container */}
            <div className="relative bg-[#313131] rounded-2xl px-20 py-12 w-[60%] shadow-lg shadow-emerald-500/10">
              {/* Update vs create wording (#317): updating an existing
                  assistant must never read as creating a new one. */}
              {task.isUpdate ? (
                <>
                  <h2 className="text-xl font-semibold text-white pr-4">
                    <Trans
                      i18nKey="knowledgeBase:confirm.updateTitle"
                      values={{ name: task.modelName, fileCount: task.paths?.length || 0 }}
                      components={{ b: <span className="font-bold" /> }}
                    />
                  </h2>
                  <p className="mt-1 text-gray-300">{t("knowledgeBase:confirm.updateBody")}</p>
                </>
              ) : (
                <>
                  <h2 className="text-xl font-semibold text-white pr-4">
                    <Trans
                      i18nKey="knowledgeBase:confirm.createTitle"
                      values={{ name: task.modelName }}
                      components={{ b: <span className="font-bold" /> }}
                    />
                  </h2>
                  <p className="mt-1 text-gray-300">
                    {t("knowledgeBase:confirm.createBody", {
                      fileCount: task.paths?.length || 0,
                    })}
                  </p>
                </>
              )}

              <div className="mt-4 flex justify-start gap-4">
                <button
                  onClick={cancelConfirm}
                  className="px-4 py-1 border border-red-500 text-red-500 rounded-full hover:bg-red-500/10 transition-shadow shadow-none hover:shadow-lg"
                >
                  {t("common:actions.cancel")}
                </button>
                <button
                  onClick={startCreation}
                  className="px-4 py-2 border border-emerald-500 text-emerald-500 rounded-full hover:bg-emerald-500/10 transition-shadow shadow-none hover:shadow-lg"
                >
                  {task.isUpdate
                    ? t("knowledgeBase:confirm.update")
                    : t("knowledgeBase:confirm.create")}
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}

      {/* Bottom-left spinner - only show when creating (after API call succeeds) */}
      {showSpinner &&
        (status === "pending" || status === "running") &&
        ReactDOM.createPortal(
          // Decorative only: it overlaps the rail's Settings entry, which must
          // stay clickable while a knowledge base is being built (#347).
          <div className="fixed bottom-7 left-[1.5%] pointer-events-none">
            <SpinnerDots className="w-6 h-6 text-emerald-400 animate-spin" />
          </div>,
          document.body
        )}
    </KnowledgeBaseContext.Provider>
  );
}

export const useKnowledgeBase = () => {
  const context = useContext(KnowledgeBaseContext);
  if (!context) {
    throw new Error("useKnowledgeBase must be used within a KnowledgeBaseProvider");
  }
  return context;
};
