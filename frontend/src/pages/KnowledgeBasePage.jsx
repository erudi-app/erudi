import React, { useEffect, useRef, useState } from "react";
import { HelpCircle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { inferenceTierLabel } from "../utils/inferenceTier";
import { formatNumber } from "../i18n/format";
import { useSearchParams, useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import ModelLibrary from "../components/ModelLibrary";
import InfoRow from "../components/InfoRow";
import DragDropArea from "../components/DragDropArea";
import { useKnowledgeBase } from "../contexts/KnowledgeBaseContext";
import ErrorModal from "../components/modals/ErrorModal";
import apiClient from "../services/api/client";
import EmbeddingModelGateModal from "../components/modals/EmbeddingModelGateModal";
import { GATE, gateStateFromStatus, shouldPoll, isGateBlocking } from "../utils/embeddingGate";
import { isKbAssistant } from "../utils/modelWeights";
import { createLogger } from "../utils/logger";

const log = createLogger("KnowledgeBasePage");

export default function KnowledgeBasePage() {
  const { t } = useTranslation();
  const { open: openKnowledgeBase, isCreating, isStarting } = useKnowledgeBase();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [errorMessage, setErrorMessage] = useState("");
  const [isValidated, setIsValidated] = useState(false);

  // Inference rating readout. The state holds a status code plus the raw
  // backend values; the displayed placeholders ("fetching…", "N/A", …) are
  // resolved at render time so they follow the app language (#385).
  const [rating, setRating] = useState({ status: "fetching", score: null, label: null });
  // The backend label is an English tier name; show the same translated tier
  // the Models page shows (#387), never the raw label.
  const ratingLabel =
    rating.status === "ready"
      ? inferenceTierLabel(rating.label) || t("knowledgeBase:page.rating.notAvailable")
      : t(`knowledgeBase:page.rating.${rating.status}`);
  const ratingScore =
    rating.status === "ready"
      ? rating.score
        ? t("knowledgeBase:page.rating.score", {
            score: formatNumber(rating.score, { maximumFractionDigits: 1 }),
          })
        : t("knowledgeBase:page.rating.notAvailable")
      : t(`knowledgeBase:page.rating.${rating.status}`);

  const [selectedModel, setSelectedModel] = useState(null);
  const [modelName, setModelName] = useState("");
  const [description, setDescription] = useState("");
  const [paths, setPaths] = useState([]);
  const [models, setModels] = useState([]);
  // Bumped once a submission completes, so DragDropArea drops the staged
  // file list it owns along with the parent's own state.
  const [formResetKey, setFormResetKey] = useState(0);

  // --- Embedding-model gate (#146): the KB needs the e5 model on disk. ---
  const [gateState, setGateState] = useState(GATE.CHECKING);
  // True after a failed gate check has been logged, until the next success.
  const gateCheckFailedRef = useRef(false);
  const [gateError, setGateError] = useState(null);

  // Handle files dropped from DragDropArea
  const addDroppedFiles = (newPathObjects) => {
    log.log("Received files from drag-drop area", newPathObjects);

    // Handle complete replacement of the file list (for when files are removed)
    // or addition of new files (for when files are added)
    setPaths(() => {
      const newPaths = newPathObjects.map((pathObj) => pathObj.path || pathObj);
      log.log("Setting file paths", newPaths);
      return Array.from(new Set(newPaths)); // Remove duplicates but don't merge with previous
    });
  };

  const closeErrorModal = () => {
    setErrorMessage("");
  };

  /* helper to determine bullet or icon for rating field */
  const getRatingBulletOrIcon = ({ status, label }) => {
    // Still fetching: show question mark icon
    if (status === "fetching") {
      return {
        type: "icon",
        value: <HelpCircle className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400" />,
      };
    }

    // Color code based on the backend's rating label (not user-facing copy:
    // the label itself is rendered as received).
    if (label === "Amazing" || label === "Excellent" || label === "Very High") {
      return { type: "bullet", value: "bg-emerald-400" };
    } else if (label === "Good" || label === "Medium" || label === "Bad" || label === "High") {
      return { type: "bullet", value: "bg-orange-400" };
    } else {
      return { type: "bullet", value: "bg-red-500" };
    }
  };

  const submitTrainForm = async () => {
    log.log("Submitting knowledge base form", {
      selectedModel,
      modelName,
      pathCount: paths.length,
    });

    if (!selectedModel || !modelName.trim() || paths.length === 0) {
      // A form the user has not finished: expected, not a defect.
      log.info("Knowledge base form validation failed", {
        selectedModel: !selectedModel,
        modelNameEmpty: !modelName.trim(),
        noPaths: paths.length === 0,
      });
      setErrorMessage(t("knowledgeBase:page.errors.missingFields"));
      return;
    }

    // Update vs create (#317): selecting the assistant itself means its
    // existing KB gets new documents — the backend routes on the same signal
    // (is_attached_to_kb), and the confirmation dialog must say update.
    const selected = models.find((m) => m.id === selectedModel);
    const isUpdate = isKbAssistant(selected);
    const trimmedName = modelName.trim();

    // Duplicate-name guard (#317): a second assistant with an existing local
    // model's name would be indistinguishable in every picker. Updates keep
    // the assistant's own (necessarily existing) name, so they skip the check.
    if (!isUpdate) {
      const duplicate = models.find(
        (m) => (m.name || "").trim().toLowerCase() === trimmedName.toLowerCase()
      );
      if (duplicate) {
        log.info("Duplicate assistant name rejected", { name: trimmedName });
        setErrorMessage(t("knowledgeBase:page.errors.duplicateName", { name: duplicate.name }));
        return;
      }
    }

    log.log("Validation passed, proceeding with creation");
    setErrorMessage("");

    const task = {
      paths,
      selectedModel,
      modelName: trimmedName,
      description: description.trim(),
      isUpdate,
    };

    openKnowledgeBase(task, {
      onComplete: () => {
        log.log("Knowledge base assistant created successfully");
        setIsValidated(true);
        // Reset form after a delay
        setTimeout(() => {
          setIsValidated(false);
          setPaths([]);
          setModelName("");
          setDescription("");
          // The staged file list lives inside DragDropArea, so clearing the
          // parent state is not enough: bump the key to remount it, otherwise
          // the form keeps showing files it no longer holds.
          setFormResetKey((k) => k + 1);
        }, 3000);
      },
      // UI only: the context wrote the record (it knows the job and the
      // backend's message); a second ERROR here would show one failure twice.
      onError: (error) => setErrorMessage(error),
    });
  };

  const fetchModels = () => {
    apiClient
      .get("/llms/local")
      .then((data) => {
        log.log("Fetched models", { count: data ? data.length : 0 });
        setModels(data || []);
      })
      .catch((err) => {
        log.error("Failed to fetch models", err);
        setModels([]);
      });
  };

  useEffect(() => {
    apiClient
      .get("/hardware/app_startup")
      .then((data) => {
        setRating({
          status: "ready",
          score: data.global_inference_score || null,
          label: data.global_inference_label || null,
        });
      })
      .catch((err) => {
        // The page works without the rating: degraded.
        log.warn("Failed to fetch the hardware rating", err);
        setRating({ status: "error", score: null, label: null });
      });
    fetchModels();
  }, []);

  // Applied once per distinct `model` URL param value, not on every models
  // replacement: fetchModels() (the initial load, and the Refresh icon) hands
  // the effect a new `models` array on every call, and re-running the
  // pre-fill on an unchanged param wiped out a name the user had since typed
  // (#510 follow-up).
  const appliedModelParamRef = useRef(null);

  // Handle URL parameter for model selection
  useEffect(() => {
    const modelParam = searchParams.get("model");
    if (!modelParam || models.length === 0) return;
    if (appliedModelParamRef.current === modelParam) return;

    // Find the model by name or id
    const foundModel = models.find(
      (model) =>
        model.name === modelParam ||
        model.id === modelParam ||
        model.name.toLowerCase() === modelParam.toLowerCase()
    );

    if (foundModel) {
      appliedModelParamRef.current = modelParam;
      log.log("Setting model from URL parameter", { name: foundModel.name });
      setSelectedModel(foundModel.id);
      // Update vs create (#317): only a KB assistant's own name is safe to
      // pre-fill, since the update flow keeps it. A base model's name is
      // not what the user is submitting here, and pre-filling it let a
      // name typed but never confirmed submit the base model's name
      // instead (#510).
      setModelName(isKbAssistant(foundModel) ? foundModel.name : "");
    } else {
      log.warn("Model not found for parameter", { modelParam });
    }
  }, [searchParams, models]); // Re-run when searchParams or models change

  // Handle model selection from ModelLibrary
  const handleModelSelect = (modelId) => {
    setSelectedModel(modelId);
  };

  // Handle model name change from ModelLibrary
  const handleModelNameChange = (name) => {
    setModelName(name);
  };

  // --- Embedding-model gate (#146): presence is filesystem-driven; the modal
  // blocks the KB page until the model is on disk. ---
  const refreshGateStatus = async (prev) => {
    try {
      const status = await apiClient.get("/knowledge_base/embedding-model/status");
      gateCheckFailedRef.current = false;
      setGateError(status.error || null);
      setGateState((current) => gateStateFromStatus(status, prev ?? current));
      return status;
    } catch (err) {
      // Polled every 2s while a download runs: one record per outage, not
      // one per tick. The flag resets on the next successful check.
      if (!gateCheckFailedRef.current) {
        gateCheckFailedRef.current = true;
        log.error("Embedding-model status check failed", err);
      }
      return null;
    }
  };

  // Check presence on mount; if a download is already running, enter the spinner.
  useEffect(() => {
    refreshGateStatus(GATE.CHECKING);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll only while a download is in flight (survives leaving/returning to KB).
  useEffect(() => {
    if (!shouldPoll(gateState)) return undefined;
    const id = setInterval(() => refreshGateStatus(), 2000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gateState]);

  const handleGateDownload = async () => {
    setGateState(GATE.DOWNLOADING);
    setGateError(null);
    try {
      await apiClient.post("/knowledge_base/embedding-model/download");
    } catch (err) {
      // The download the user asked for did not start.
      log.error("Embedding-model download request failed", err);
      setGateError(String(err?.message || err));
      setGateState(GATE.ERROR);
    }
  };

  // "Not now" / decline: the KB is unusable without the embedding model, so
  // leave the page entirely (back to the landing) rather than sit on a dead KB.
  const handleGateLeave = () => navigate("/erudi/models");
  // "Close" after a successful download: the model is present now, stay on the KB.
  const handleGateClose = () => setGateState(GATE.HIDDEN);

  return (
    <div className="flex h-screen bg-[#071b18]">
      <Sidebar />

      {/* The gate overlay is scoped to this <main> (relative + absolute modal):
          it blurs and blocks the KB content only, keeping the sidebar usable
          while the embedding model downloads. */}
      <main className="relative flex-1 p-4 md:p-6 lg:p-8 flex flex-col gap-4 md:gap-6 overflow-hidden">
        {isGateBlocking(gateState) && (
          <EmbeddingModelGateModal
            state={gateState}
            error={gateError}
            onDownload={handleGateDownload}
            onLeave={handleGateLeave}
            onClose={handleGateClose}
          />
        )}
        {/* Top Section: Hardware + Model Library */}
        <div className="flex flex-col lg:flex-row gap-4 md:gap-6 flex-1 min-h-0">
          <div className="relative rounded-2xl overflow-hidden shadow-xl flex-1 min-w-[340px] border border-[#385B4F] border-[0.3px] bg-[rgba(22,40,36,0.45)] flex flex-col">
            <div
              className="absolute inset-0 opacity-[11%] pointer-events-none"
              style={{
                background:
                  "linear-gradient(135deg,rgba(217,217,217,1) 0%,rgba(217,217,217,0.26) 26%,rgba(0,204,133,1) 100%)",
              }}
            />
            <div className="absolute inset-0 mix-blend-overlay pointer-events-none" />
            <div className="relative z-10 px-4 py-3 sm:px-6 sm:py-4 md:px-8 md:py-5 flex flex-col h-full overflow-hidden">
              {/* Title */}
              <h2 className="text-white text-xl sm:text-2xl md:text-3xl font-bold mb-3 md:mb-4 flex-shrink-0">
                {t("knowledgeBase:page.title")}
              </h2>

              {/* Knowledge Base description - scrollable */}
              <div className="flex-1 overflow-y-auto custom-scroll pr-2">
                <p className="text-gray-300 text-sm sm:text-base md:text-lg leading-relaxed">
                  {t("knowledgeBase:page.intro.p1")}
                  <br />
                  <br />
                  {t("knowledgeBase:page.intro.p2")}
                  <br />
                  <br />
                  {t("knowledgeBase:page.intro.p3")}
                  <br />
                  <br />
                  {t("knowledgeBase:page.intro.p4")}
                </p>
              </div>

              {/* Rating - fixed at bottom */}
              <div className="flex-shrink-0 mt-3 md:mt-4">
                <InfoRow
                  label={t("knowledgeBase:page.rating.label")}
                  isHeader={true}
                  {...(getRatingBulletOrIcon(rating).type === "bullet"
                    ? { bullet: getRatingBulletOrIcon(rating).value }
                    : { icon: getRatingBulletOrIcon(rating).value })}
                >
                  <div className="flex items-center gap-2">
                    <span>{ratingLabel}</span>
                    <span className="text-xs text-gray-400 bg-gray-800/50 px-2 py-0.5 rounded-full border border-gray-600/30">
                      {ratingScore}
                    </span>
                  </div>
                </InfoRow>
              </div>
            </div>
          </div>

          <ModelLibrary
            key={`model-library-${formResetKey}`}
            models={models}
            selectedModel={selectedModel}
            modelName={modelName}
            onModelSelect={handleModelSelect}
            onModelNameChange={handleModelNameChange}
            onRefresh={fetchModels}
          />
        </div>

        {/* Bottom Section: Dataset */}
        <div className="flex flex-col flex-1 min-h-0">
          <div className="bg-[#2B2B2B] rounded-2xl p-4 md:p-6 lg:p-8 text-white flex flex-col lg:flex-row gap-4 md:gap-6 shadow-lg h-full overflow-hidden">
            <div className="flex flex-col gap-3 md:gap-4 w-full lg:w-[44%] overflow-hidden">
              <div className="flex flex-col w-full h-full overflow-hidden">
                {/* Title */}
                <h3 className="text-white text-lg sm:text-xl md:text-2xl font-semibold mb-3 md:mb-4 text-center flex-shrink-0">
                  {t("knowledgeBase:page.form.title")}
                </h3>

                {/* Description input */}
                <textarea
                  className="w-full flex-1 bg-[#1A1A1A] text-white rounded-lg p-3 md:p-4 resize-none border border-white/10 focus:outline-none focus:ring-2 focus:ring-emerald-400/60 focus:border-emerald-400/60 transition-all placeholder-gray-400 text-sm sm:text-base"
                  placeholder={t("knowledgeBase:page.form.descriptionPlaceholder")}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </div>

              <div className="flex flex-col items-center gap-3 flex-shrink-0">
                {isValidated ? (
                  <div className="w-full text-center">
                    <div className="text-emerald-400 text-sm">
                      {t("knowledgeBase:page.form.success")}
                    </div>
                    <div className="inline-flex items-center gap-2 py-3"></div>
                  </div>
                ) : (
                  <button
                    className="py-2 md:py-3 px-6 md:px-8 rounded-full bg-emerald-500 text-white font-semibold shadow-lg hover:bg-emerald-400 transition disabled:opacity-50 text-sm sm:text-base"
                    onClick={() => {
                      submitTrainForm();
                    }}
                    disabled={isCreating || isStarting}
                  >
                    {isCreating
                      ? t("knowledgeBase:page.form.creating")
                      : t("knowledgeBase:page.form.create")}
                  </button>
                )}

                {/* Error Modal */}
                <ErrorModal errorMessage={errorMessage} onClose={closeErrorModal} />
              </div>
            </div>

            <div className="w-full lg:w-[56%] h-full overflow-hidden">
              <DragDropArea key={`drag-drop-${formResetKey}`} onFilesAdded={addDroppedFiles} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
