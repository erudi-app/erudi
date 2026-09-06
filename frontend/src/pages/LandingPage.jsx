import React, { useState, useEffect, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Trans, useTranslation } from "react-i18next";
import Sidebar from "../components/Sidebar";
import ModelCollapsibleSection from "../components/ModelCollapsibleSection";
import ModelCard from "../components/ModelCard";
import ExploreModelCard from "../components/ExploreModelCard";
import MachineReadout from "../components/MachineReadout";
import HuggingFaceSearchPanel from "../components/HuggingFaceSearchPanel";
import CategorySections from "../components/CategorySections";
import CatalogFilters from "../components/CatalogFilters";
import CatalogSearch from "../components/CatalogSearch";
import ExploreIndex from "../components/ExploreIndex";
import ConnectionStatus from "../components/ConnectionStatus";
import ModelInfoModal from "../components/modals/ModelInfoModal";
import DeleteModelModal from "../components/modals/DeleteModelModal";
import MessageModal from "../components/modals/MessageModal";
import { useDownloadModal } from "../contexts/DownloadModalContext";
import HardwareLoadingPopup from "../components/LoadingPopup";
import { RefreshCcw } from "lucide-react";
import WelcomeModal from "../components/modals/WelcomeModal";
import logoErudi from "../assets/images/logos/logoerudifinal.png";
import { API_BASE_URL } from "../config/api";
import apiClient, { tracedFetch } from "../services/api/client";
import { transformAppStartupInfo } from "../utils/hardwareTransform";
import { createLogger } from "../utils/logger";
import { splitByBase, installedRepoKeys, modelRepoKey } from "../utils/modelCatalog";
import { searchCatalog } from "../utils/catalogSearch";
import { rankByFit, pickFlagships, applyCatalogFilters } from "../utils/hardwareFit";
import useDebouncedValue from "../shared/hooks/useDebouncedValue";
import { isTestedModel } from "../utils/testedModels";
import { isKbAssistant, hasMissingWeights, findBaseModelName } from "../utils/modelWeights";
import { fetchDeleteDependents, parseConflictDependents } from "../utils/deleteGuard";
import { formatNumber } from "../i18n/format";

export default function LandingPage() {
  const log = createLogger("LandingPage");
  const { t } = useTranslation();

  const { open, completionCount } = useDownloadModal();
  const navigate = useNavigate();
  const [showWelcome, setShowWelcome] = useState(false);
  const [showLoadingPopup, setShowLoadingPopup] = useState(false);
  const [hardwareInfo, setHardwareInfo] = useState(null);
  const [machineDetail, setMachineDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [localModels, setLocalModels] = useState([]);
  const [remoteModels, setRemoteModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [selectedModelInfo, setSelectedModelInfo] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [deleteConfirmation, setDeleteConfirmation] = useState({
    show: false,
    model: null,
    dependents: null,
  });
  const [brainSidebarCollapsed, setBrainSidebarCollapsed] = useState(false);
  const [communityOpen, setCommunityOpen] = useState(false);
  const [filters, setFilters] = useState({ size: "any", fitOnly: false });
  // Offline catalog search (#380): the box is controlled on every keystroke,
  // the matching runs a beat later on the debounced value. Clearing is
  // immediate (an empty box never waits for the debounce).
  const [catalogInput, setCatalogInput] = useState("");
  const debouncedCatalogQuery = useDebouncedValue(catalogInput.trim(), 150);
  const catalogQuery = catalogInput.trim() ? debouncedCatalogQuery : "";
  // Term handed to the Hugging Face panel from an empty catalog result; `seq`
  // makes handing the same term over twice a new event.
  const [hfHandoff, setHfHandoff] = useState(null);
  const [online, setOnline] = useState(
    () => typeof navigator === "undefined" || navigator.onLine !== false
  );
  const localModelsRef = useRef(null);

  // Same signal the Hugging Face panel uses for its offline guard, kept live so
  // the "search Hugging Face instead" escalation appears/disappears with it.
  useEffect(() => {
    const update = () => setOnline(navigator.onLine !== false);
    window.addEventListener("online", update);
    window.addEventListener("offline", update);
    return () => {
      window.removeEventListener("online", update);
      window.removeEventListener("offline", update);
    };
  }, []);

  // Helper function to parse model metadata
  const parseMetadata = (metadataString) => {
    if (!metadataString) {
      return {};
    }
    try {
      const lines = metadataString.split("\n");
      const metadata = {};
      lines.forEach((line) => {
        const trimmedLine = line.trim();
        if (trimmedLine.includes(":")) {
          const [key, ...valueParts] = trimmedLine.split(":");
          const value = valueParts.join(":").trim();
          const cleanKey = key.trim().toLowerCase().replace(/\s+/g, "_");
          metadata[cleanKey] = value;
        }
      });
      return metadata;
    } catch (error) {
      return {};
    }
  };

  const unknown = t("common:status.unknown");

  const transformRemote = (model) => {
    const metadata = parseMetadata(model.model_metadata);
    return {
      id: model.id,
      name: model.name,
      size: metadata.size || unknown,
      // Fields the details modal reads, derived from the parsed metadata.
      parameters: metadata.parameters || (model.param_size ? `${model.param_size}B` : unknown),
      downloads: metadata.downloads || unknown,
      likes: metadata.likes || unknown,
      author: metadata.author || unknown,
      library: metadata.library || unknown,
      pipeline: metadata.pipeline || unknown,
      lastUpdate: metadata.last_modified || unknown,
      description: model.description,
      runnable: model.runnable !== false,
      is_base: model.is_base === true,
      category: model.category || "general",
      type: model.type,
      param_size: model.param_size,
      // Measured download size when the backend has one (#387); the size
      // line is formatted at render time from these fields.
      artifact_size_bytes: model.artifact_size_bytes,
      link: model.link,
      quantized: model.quantized,
      // Resolved sampling defaults (#388): the details modal reads `source`
      // to say when the publisher gives no recommendation.
      sampling_defaults: model.sampling_defaults,
      metadata,
      rawMetadata: model.model_metadata,
    };
  };

  const transformLocal = (model) => {
    const metadata = parseMetadata(model.model_metadata);
    return {
      id: model.id,
      name: model.name,
      size: metadata.size || unknown,
      parameters: metadata.parameters || unknown,
      // Measured size in billions; the very-small-model note (#381) reads it
      // before falling back to the metadata string above.
      param_size: model.param_size,
      quantized: model.quantized,
      // Measured download size when the backend has one (#387); the size
      // line is formatted at render time from these fields.
      artifact_size_bytes: model.artifact_size_bytes,
      lastUpdate: metadata.last_modified || unknown,
      isOnline: false,
      description: model.description,
      // Orphan-model UX (#225/#208): `link` ties a KB assistant to the base
      // model whose weights it uses; kb_id/is_attached_to_kb identify assistant
      // rows; weights_available === false marks an orphan (weights deleted).
      link: model.link,
      kb_id: model.kb_id ?? null,
      is_attached_to_kb: model.is_attached_to_kb === true,
      weights_available: model.weights_available,
      // Resolved sampling defaults (#388), read by the details modal.
      sampling_defaults: model.sampling_defaults,
      metadata,
      rawMetadata: model.model_metadata,
    };
  };

  // Stable fetcher for both models lists, reusable outside the mount effect
  // (e.g. to silently refresh a stale catalog after a failed download start, #167).
  const refreshCatalog = useCallback(async () => {
    setModelsLoading(true);
    try {
      const localResponse = await tracedFetch(`${API_BASE_URL}/llms/local`);
      if (localResponse.ok) {
        const localData = await localResponse.json();
        setLocalModels(localData.map(transformLocal));
      }
      const remoteResponse = await tracedFetch(`${API_BASE_URL}/llms/remote`);
      if (remoteResponse.ok) {
        const remoteData = await remoteResponse.json();
        setRemoteModels(remoteData.map(transformRemote));
      }
    } catch (error) {
      log.error("Error fetching models:", error);
    } finally {
      setModelsLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const fetchWelcomePopupStatus = async () => {
      try {
        const data = await apiClient.get("/startup/welcome-popup");
        setShowWelcome(!data.has_already_displayed);
      } catch (error) {
        // The popup is simply not shown: degraded.
        log.warn("Failed to fetch the welcome popup status", error);
      }
    };

    const fetchHardwareEvaluation = async () => {
      try {
        const data = await apiClient.get("/hardware/app_startup");
        setHardwareInfo(transformAppStartupInfo(data));
      } catch (error) {
        // The hero and the recommendations depend on this answer.
        log.error("Failed to fetch the hardware evaluation", error);
        setHardwareInfo({
          backend_type: "unknown",
          error: t("landing:messages.hardwareEvaluationFailed"),
        });
      } finally {
        setLoading(false);
      }
    };

    // Richer hardware detail for the machine readout (chip, memory, GPU cores).
    const fetchMachineDetail = async () => {
      try {
        const response = await tracedFetch(`${API_BASE_URL}/hardware/detailed`);
        if (response.ok) {
          const data = await response.json();
          setMachineDetail(data.hardware || null);
        }
      } catch (error) {
        // The readout falls back to the summary: degraded.
        log.warn("Failed to fetch the hardware detail", error);
      }
    };

    fetchWelcomePopupStatus();
    fetchHardwareEvaluation();
    fetchMachineDetail();
    refreshCatalog();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshCatalog]);

  const closeWelcome = () => {
    if (loading) {
      setShowLoadingPopup(true);
      return;
    }
    setShowWelcome(false);
  };

  const closeLoadingOnly = () => setShowLoadingPopup(false);

  const handleMainPageRefresh = async () => {
    await reloadLocalModels();
  };

  const reloadLocalModels = async () => {
    setModelsLoading(true);
    try {
      const res = await tracedFetch(`${API_BASE_URL}/llms/local`);
      if (res.ok) {
        const localData = await res.json();
        setLocalModels(localData.map(transformLocal));
      } else {
        setErrorMessage(t("landing:messages.fetchLocalModelsFailed"));
      }
    } catch (err) {
      setErrorMessage(t("landing:messages.fetchLocalModelsFailed"));
    } finally {
      await new Promise((resolve) => setTimeout(resolve, 600));
      setModelsLoading(false);
    }
  };

  // A completed download bumps the context's completionCount, whichever entry
  // point started it and even if this page mounted after the download began.
  // Refresh the installed lists on every tick so the models show up without a
  // manual reload — the per-download onComplete callback alone misses the cases
  // where the ref was overwritten or the registering page had unmounted (#205).
  useEffect(() => {
    if (!completionCount) {
      return;
    }
    reloadLocalModels();
    if (localModelsRef.current) {
      localModelsRef.current.reloadLocalModels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [completionCount]);

  // Derived: Base vs Community (backend is_base), hardware-fit window, and the
  // best-fitting base models for the recommendation rail (#122 redesign).
  const { base: baseModels, community: communityModels } = splitByBase(remoteModels);
  const range = hardwareInfo
    ? { min: hardwareInfo.recommended_param_min, max: hardwareInfo.recommended_param_max }
    : null;
  // Recommended row leads with team-certified models (the BadgeCheck picks),
  // then fills with hardware-fit flagships — certified models are pushed to the
  // front here instead of getting a section of their own (#tested).
  const flagships = pickFlagships(baseModels, range, 3);
  const certifiedFit = rankByFit(baseModels.filter(isTestedModel), range);
  const recommended = [...certifiedFit, ...flagships]
    .filter((m, i, arr) => arr.findIndex((x) => (x.id ?? x.link) === (m.id ?? m.link)) === i)
    .slice(0, 3);
  const filteredBase = applyCatalogFilters(baseModels, filters, range);
  const filteredCommunity = applyCatalogFilters(communityModels, filters, range);
  const filtersActive = filters.size !== "any" || filters.fitOnly;
  // Search hits over the whole bundled catalog (base first, then community),
  // AND-ed with the size / fit filters above; null while no query is active.
  const catalogHits = catalogQuery
    ? searchCatalog([...filteredBase, ...filteredCommunity], catalogQuery)
    : null;

  // Catalog cards for models already on disk (#348). The local row and the
  // catalog row are separate records joined only by the Hugging Face repo id
  // both carry in their metadata, so nothing linked them before and an
  // installed model kept an enabled Download button in every browse section.
  const installedKeys = installedRepoKeys(localModels);
  const isInstalled = (model) => {
    const key = modelRepoKey(model);
    return key ? installedKeys.has(key) : false;
  };

  // Installed models an orphaned assistant can be re-bound to: local,
  // non-assistant, weights still on disk (#225).
  const rebindTargets = localModels.filter((m) => !isKbAssistant(m) && !hasMissingWeights(m));

  const machine = {
    chip: machineDetail?.mlx_chip_model
      ? `Apple ${machineDetail.mlx_chip_model}`
      : machineDetail?.gpu_name || machineDetail?.cpu_model || t("landing:machine.fallbackChip"),
    backend: (hardwareInfo?.backend_type || "").toUpperCase(),
    memoryGb: machineDetail?.total_memory_gb ? Math.round(machineDetail.total_memory_gb) : null,
    gpuCores: machineDetail?.mlx_gpu_cores || null,
    bandwidth: machineDetail?.memory_bandwidth_gbs
      ? Math.round(machineDetail.memory_bandwidth_gbs)
      : null,
    // Only the CUDA hardware branch carries vram_total_gb; null everywhere else
    // so MachineReadout only renders the VRAM stat on NVIDIA machines (#202).
    vramGb: machineDetail?.vram_total_gb ? Math.round(machineDetail.vram_total_gb) : null,
    inferenceLabel: hardwareInfo?.global_inference_label,
    inferenceScore: hardwareInfo?.global_inference_score,
    range,
  };

  const handleDownload = (model) => {
    if (open) {
      open(model, {
        onComplete: async () => {
          await reloadLocalModels();
          if (localModelsRef.current) {
            localModelsRef.current.reloadLocalModels();
          }
        },
        onError: () => {
          // A failed start means the page data may be stale (e.g. the catalog id
          // was deleted), so refresh silently and let the user re-click (#167).
          // The failure itself is shown by the download widget (#292): no
          // second dialog on top of it.
          refreshCatalog().catch(() => {});
        },
      });
    }
  };

  const handleInfo = (model) => setSelectedModelInfo(model);
  const handleChat = (model) => navigate(`/erudi/chat?model=${encodeURIComponent(model.name)}`);
  const handleKnowledgeBase = (model) =>
    navigate(`/erudi/attach_knowledge_base?model=${encodeURIComponent(model.name)}`);
  // Guarded base delete (#225): pre-check the dependents endpoint so the
  // confirmation dialog can list what the deletion orphans. Best-effort — if
  // the pre-check fails the plain dialog opens and the DELETE's 409 below
  // remains the safety net. The guard is keyed on actual dependency, never on
  // the shared weights link (#317): deleting a KB assistant is a direct 200
  // that frees nothing (the weights belong to its base), so assistants skip
  // the pre-check entirely and always get the plain confirm.
  const handleDelete = async (model) => {
    const dependents = await fetchDeleteDependents(model);
    setDeleteConfirmation({ show: true, model, dependents });
  };

  const confirmDelete = async () => {
    if (!deleteConfirmation.model) {
      return;
    }
    const modelToDelete = deleteConfirmation.model;
    // Confirming a dialog that listed dependents means "Delete anyway":
    // the base goes, its assistants stay (orphaned) and conversations are kept.
    const orphanDependents = Boolean(deleteConfirmation.dependents?.assistants?.length);
    setDeleteConfirmation({ show: false, model: null, dependents: null });
    try {
      const url = orphanDependents
        ? `${API_BASE_URL}/llms/${modelToDelete.id}?orphan_dependents=true`
        : `${API_BASE_URL}/llms/${modelToDelete.id}`;
      const response = await tracedFetch(url, {
        method: "DELETE",
      });
      if (response.ok) {
        setSuccessMessage(t("landing:messages.modelDeleted", { name: modelToDelete.name }));
        await reloadLocalModels();
        if (localModelsRef.current) {
          localModelsRef.current.reloadLocalModels();
        }
      } else if (response.status === 409) {
        // Safety net: the pre-check missed dependents (raced or failed). The
        // 409 payload carries the same dependents shape — reopen the dialog
        // with it instead of surfacing an error.
        const detail = await parseConflictDependents(response);
        if (detail) {
          setDeleteConfirmation({ show: true, model: modelToDelete, dependents: detail });
          return;
        }
        throw new Error(`Failed to delete model: ${response.status}`);
      } else {
        throw new Error(`Failed to delete model: ${response.status}`);
      }
    } catch (error) {
      log.error("Failed to delete model:", error);
      setErrorMessage(t("landing:messages.deleteModelFailed"));
    }
  };

  const cancelDelete = () => setDeleteConfirmation({ show: false, model: null, dependents: null });

  // Re-bind an orphaned KB assistant to another installed base model (#225):
  // POST rebind, then refresh so the card leaves its "weights missing" state.
  const handleRebind = async (assistant, target) => {
    try {
      const response = await tracedFetch(`${API_BASE_URL}/llms/${assistant.id}/rebind`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_base_llm_id: target.id }),
      });
      if (!response.ok) {
        throw new Error(`Failed to rebind assistant: ${response.status}`);
      }
      setSuccessMessage(
        t("landing:messages.assistantRebound", { assistant: assistant.name, target: target.name })
      );
      await reloadLocalModels();
      if (localModelsRef.current) {
        localModelsRef.current.reloadLocalModels();
      }
    } catch (error) {
      log.error("Failed to rebind assistant:", error);
      setErrorMessage(t("landing:messages.rebindFailed"));
    }
  };
  const handleToggleBrainSidebar = () => setBrainSidebarCollapsed(!brainSidebarCollapsed);

  // Left-rail Explore index scrolls the main panel to a section.
  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  // Nothing in the bundled catalog matched: escalate the same term to the
  // Hugging Face panel, which is the "look beyond the curated list" step.
  const handOffToHuggingFace = (term) => {
    setHfHandoff((prev) => ({ term, seq: (prev?.seq ?? 0) + 1 }));
    scrollToSection("explore-search");
  };

  return (
    <div className="flex h-screen">
      <Sidebar
        showBrainCollapsible={true}
        onToggleBrainSidebar={handleToggleBrainSidebar}
        brainCollapsed={brainSidebarCollapsed}
      />

      <aside
        className={`${brainSidebarCollapsed ? "w-0 opacity-0 overflow-hidden" : "w-64 opacity-100 p-6 overflow-visible"} bg-[#272727] text-white flex flex-col transition-all duration-300`}
      >
        <div className="flex items-center justify-start mb-6 flex-shrink-0">
          <img
            src={logoErudi}
            alt="Erudi"
            className="h-[40px] ml-2 w-auto cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => setShowWelcome(true)}
            onError={(e) => log.warn("Failed to load the logo", e.target.src)}
          />
        </div>
        <div className="mb-6 flex-shrink-0">
          <ModelCollapsibleSection
            kind="local"
            ref={localModelsRef}
            onLocalModelRefresh={handleMainPageRefresh}
          />
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto custom-scroll">
          <ExploreIndex
            models={filteredBase}
            communityCount={filteredCommunity.length}
            hasRecommended={recommended.length > 0}
            loading={modelsLoading}
            onJump={scrollToSection}
          />
        </div>
        <div className="flex-shrink-0">
          <ConnectionStatus />
        </div>
      </aside>

      {/* Main explore panel */}
      <main className="flex-1 bg-[var(--canvas)] relative custom-scroll overflow-auto">
        <div className="mx-auto max-w-6xl px-8 py-8 space-y-9">
          {/* Hero: machine readout — the spine of the panel */}
          <MachineReadout machine={machine} loading={loading} />

          {/* Local models */}
          <section>
            <div className="flex items-center justify-between mb-4">
              <span className="eyebrow">{t("landing:installed.eyebrow")}</span>
              <button
                onClick={() => reloadLocalModels()}
                title={t("landing:installed.refreshTitle")}
                className="text-[var(--ink-dim)] hover:text-[var(--ink)] transition-colors"
              >
                <RefreshCcw className="w-4 h-4" />
              </button>
            </div>
            {modelsLoading ? (
              <div className="flex items-center gap-2 text-[var(--ink-faint)] mono text-xs py-6">
                <span className="w-2 h-2 rounded-full bg-[var(--fit-good)] animate-pulse" />
                {t("landing:installed.loading")}
              </div>
            ) : localModels.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
                {localModels.map((model) => (
                  <ModelCard
                    key={model.id}
                    model={model}
                    type="local"
                    onChat={handleChat}
                    onInfo={handleInfo}
                    onKnowledgeBase={handleKnowledgeBase}
                    onDelete={handleDelete}
                    baseModelName={
                      isKbAssistant(model) ? findBaseModelName(model, localModels) : null
                    }
                    rebindTargets={rebindTargets}
                    onRebind={handleRebind}
                  />
                ))}
              </div>
            ) : (
              <p className="text-[var(--ink-dim)] text-sm">
                <Trans
                  i18nKey="landing:installed.empty"
                  values={{
                    range: range
                      ? t("landing:installed.range", {
                          min: formatNumber(range.min),
                          max: formatNumber(range.max),
                        })
                      : t("landing:installed.rangeFallback"),
                  }}
                  components={{ fit: <span className="mono text-[var(--fit-good)]" /> }}
                />
              </p>
            )}
          </section>

          {/* Recommended for your machine — flagship, instruct-only picks */}
          {recommended.length > 0 && (
            <section id="explore-recommended" className="rise scroll-mt-6">
              <span className="eyebrow !text-[var(--fit-good)]">
                {t("landing:recommended.eyebrow")}
              </span>
              <p className="text-[13px] text-[var(--ink-dim)] mt-1.5 mb-4">
                {t("landing:recommended.description", { chip: machine.chip })}
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
                {recommended.map((model) => (
                  <ExploreModelCard
                    key={`rec-${model.id ?? model.link}`}
                    model={model}
                    range={range}
                    onDownload={handleDownload}
                    onInfo={handleInfo}
                    installed={isInstalled(model)}
                  />
                ))}
              </div>
            </section>
          )}

          {/* Live Hugging Face search — the research tool */}
          <div id="explore-search" className="scroll-mt-6">
            <HuggingFaceSearchPanel
              range={range}
              onDownload={handleDownload}
              onInfo={handleInfo}
              isInstalled={isInstalled}
              handoff={hfHandoff}
            />
          </div>

          {/* Browse by capability — with the offline catalog search (#380) */}
          <section id="explore-browse" className="scroll-mt-6">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <span className="eyebrow">{t("landing:browse.eyebrow")}</span>
              <CatalogFilters value={filters} onChange={setFilters} hasRange={!!range} />
            </div>
            <div className="mb-5">
              <CatalogSearch value={catalogInput} onChange={setCatalogInput} />
            </div>
            {catalogHits ? (
              catalogHits.length > 0 ? (
                <>
                  <div className="flex items-center gap-3 mb-3 flex-wrap">
                    <span className="eyebrow">
                      {t("models:catalogSearch.resultsFor", {
                        count: catalogHits.length,
                        term: catalogQuery,
                      })}
                    </span>
                    <span className="h-px flex-1 bg-white/10" />
                    <button
                      onClick={() => setCatalogInput("")}
                      className="mono text-[11px] text-[var(--ink-dim)] hover:text-[var(--ink)] transition-colors"
                    >
                      {t("common:actions.clear")}
                    </button>
                  </div>
                  <div
                    data-testid="catalog-search-results"
                    className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3"
                  >
                    {catalogHits.map((model) => (
                      <ExploreModelCard
                        key={`search-${model.id ?? model.link}`}
                        model={model}
                        range={range}
                        onDownload={handleDownload}
                        onInfo={handleInfo}
                        installed={isInstalled(model)}
                      />
                    ))}
                  </div>
                </>
              ) : (
                <div className="py-8 text-center space-y-3">
                  <p className="text-[var(--ink-dim)] text-sm">
                    {t("models:catalogSearch.noMatch", { term: catalogQuery })}
                  </p>
                  {filtersActive && (
                    <p className="text-[var(--ink-faint)] text-sm">{t("landing:browse.noMatch")}</p>
                  )}
                  {online ? (
                    <button
                      onClick={() => handOffToHuggingFace(catalogQuery)}
                      className="mono text-[11px] rounded-full border border-[var(--fit-good)] text-[var(--fit-good)] bg-[var(--fit-good)]/10 px-3 py-1.5 transition-[filter] hover:brightness-110"
                    >
                      {t("models:catalogSearch.searchHuggingFaceInstead", { term: catalogQuery })}
                    </button>
                  ) : (
                    <p className="mono text-[11px] text-[var(--ink-faint)]">
                      {t("models:search.offline")}
                    </p>
                  )}
                </div>
              )
            ) : filtersActive && filteredBase.length === 0 ? (
              <p className="text-[var(--ink-dim)] text-sm py-8 text-center">
                {t("landing:browse.noMatch")}
              </p>
            ) : (
              <CategorySections
                models={filteredBase}
                range={range}
                loading={modelsLoading}
                onDownload={handleDownload}
                onInfo={handleInfo}
                isInstalled={isInstalled}
              />
            )}
          </section>

          {/* Community fine-tunes — collapsed by default to keep the panel calm.
              Hidden while a catalog query is active: the results grid above
              already covers these rows, so the section would only duplicate
              them under the results (or under the empty state). */}
          {!catalogHits && filteredCommunity.length > 0 && (
            <section id="explore-community" className="scroll-mt-6">
              <button
                className="flex items-center gap-3 w-full text-left mb-4 group"
                onClick={() => setCommunityOpen((o) => !o)}
              >
                <span className="eyebrow group-hover:text-[var(--ink)] transition-colors">
                  {t("landing:community.title")}
                </span>
                <span className="mono text-[11px] text-[var(--ink-faint)]">
                  {formatNumber(filteredCommunity.length)}
                </span>
                <span className="h-px flex-1 bg-white/10" />
                <span className="mono text-[11px] text-[var(--ink-dim)] group-hover:text-[var(--fit-good)] transition-colors">
                  {communityOpen ? t("landing:community.hide") : t("landing:community.showAll")}
                </span>
              </button>
              {communityOpen && (
                <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 max-h-[640px] overflow-y-auto custom-scroll pr-1">
                  {rankByFit(filteredCommunity, range).map((model) => (
                    <ExploreModelCard
                      key={`com-${model.id ?? model.link}`}
                      model={model}
                      range={range}
                      onDownload={handleDownload}
                      onInfo={handleInfo}
                      installed={isInstalled(model)}
                    />
                  ))}
                </div>
              )}
            </section>
          )}
        </div>
      </main>

      <WelcomeModal
        isOpen={showWelcome}
        onClose={closeWelcome}
        hardwareInfo={hardwareInfo}
        loading={loading}
      />
      <ModelInfoModal
        modelInfo={selectedModelInfo}
        isOpen={!!selectedModelInfo}
        onClose={() => setSelectedModelInfo(null)}
        onDownload={handleDownload}
        installed={!!selectedModelInfo && isInstalled(selectedModelInfo)}
      />
      <DeleteModelModal
        isOpen={deleteConfirmation.show}
        model={deleteConfirmation.model}
        dependents={deleteConfirmation.dependents}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
      />
      <MessageModal
        isOpen={!!successMessage}
        title={t("landing:messages.successTitle")}
        message={successMessage}
        type="success"
        onClose={() => setSuccessMessage("")}
      />
      <MessageModal
        isOpen={!!errorMessage}
        title={t("common:status.error")}
        message={errorMessage}
        type="error"
        onClose={() => setErrorMessage("")}
      />
      <HardwareLoadingPopup show={showLoadingPopup} loading={loading} onClose={closeLoadingOnly} />
    </div>
  );
}
