// src/components/CollapsibleSection.jsx
import React, { useState, useEffect, forwardRef, useImperativeHandle } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import Tooltip from "./Tooltip";
import DeleteModelModal from "./modals/DeleteModelModal";
import MessageModal from "./modals/MessageModal";
import { ChevronDown, RefreshCcw, HelpCircle, Trash2, Database, Globe, Search } from "lucide-react";
import { useDownloadModal } from "../contexts/DownloadModalContext";
import { API_BASE_URL } from "../config/api";
import { tracedFetch } from "../services/api/client";
import { fetchDeleteDependents, parseConflictDependents } from "../utils/deleteGuard";
import { createLogger } from "../utils/logger";
const log = createLogger("ModelCollapsibleSection");

// Icon mapping for different section kinds
const getIconForSection = (kind) => {
  switch (kind) {
    case "remote":
      return <Globe className="w-5 h-5 font-bold text-white" />;
    case "local":
    default:
      return <Database className="w-5 h-5 font-bold text-white" />;
  }
};

const CollapsibleSection = forwardRef(
  ({ kind = "local", onLocalModelRefresh, hasSearch = false }, ref) => {
    const { t } = useTranslation();
    const isLocal = kind === "local";
    const title = isLocal ? t("landing:sidebar.localModels") : t("landing:sidebar.remoteModels");
    const [openSection, setOpenSection] = useState(true);
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState("");
    const [deleteConfirmation, setDeleteConfirmation] = useState({
      show: false,
      model: null,
      dependents: null,
    });
    const [successMessage, setSuccessMessage] = useState("");
    const [searchTerm, setSearchTerm] = useState("");

    const { open: openDownload } = useDownloadModal();

    // Expose reloadLocalModels to parent via ref
    useImperativeHandle(ref, () => ({
      reloadLocalModels,
    }));

    // TooltipIcon component - Simple CSS-based tooltip
    const TooltipIcon = () => {
      const tooltipText = isLocal
        ? t("landing:sidebar.localTooltip")
        : t("landing:sidebar.remoteTooltip");

      return (
        <Tooltip content={tooltipText} side="right" width="w-80">
          <HelpCircle className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400 hover:text-emerald-400 transition-colors cursor-help" />
        </Tooltip>
      );
    };

    // fetch models
    useEffect(() => {
      async function fetchModels() {
        setLoading(true);
        try {
          const url = isLocal ? `${API_BASE_URL}/llms/local` : `${API_BASE_URL}/llms/remote`;
          const res = await tracedFetch(url);
          if (res.ok) {
            setModels(await res.json());
          }
        } catch (err) {
          log.error("Failed to fetch models:", err);
          setErrorMessage(t("landing:messages.fetchModelsFailed"));
        } finally {
          await new Promise((resolve) => setTimeout(resolve, 1000));
          setLoading(false);
        }
      }
      fetchModels();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isLocal]);

    const reloadLocalModels = async () => {
      setLoading(true);
      try {
        const url = `${API_BASE_URL}/llms/local`;
        const res = await tracedFetch(url);
        if (res.ok) {
          setModels(await res.json());
        } else {
          setErrorMessage(t("landing:messages.fetchLocalModelsFailed"));
        }
      } catch (err) {
        log.error("Failed to fetch local models:", err);
        setErrorMessage(t("landing:messages.fetchLocalModelsFailed"));
      } finally {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        setLoading(false);
      }
    };

    const loadLocalModelsAfterDownload = () => {
      // Si on est dans la section Remote Models et qu'on a un callback pour recharger les locaux
      if (!isLocal && onLocalModelRefresh) {
        onLocalModelRefresh();
      }
    };

    const handleModelClick = (model) => {
      if (model.runnable === false) {
        return; // not offered on this hardware — the row is rendered disabled
      }
      setErrorMessage("");
      // Failures and cancellations are reported by the download widget itself
      // (#292); this rail only needs to refresh once the weights are in.
      openDownload(model, { onComplete: loadLocalModelsAfterDownload });
    };

    // Guarded base delete (#317): the rail runs the same flow as the installed
    // cards — pre-check the dependents so the dialog can list what the deletion
    // orphans, instead of a bare DELETE whose 409 used to dead-end here.
    const handleDeleteClick = async (e, model) => {
      e.stopPropagation();
      const dependents = await fetchDeleteDependents(model);
      setDeleteConfirmation({ show: true, model, dependents });
    };

    const confirmDelete = async () => {
      if (!deleteConfirmation.model) {
        return;
      }

      // Store model reference and close modal immediately to prevent double-clicks
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
          // Reload models in this component
          await reloadLocalModels();
          // Also refresh the main page local models
          if (onLocalModelRefresh) {
            onLocalModelRefresh();
          }
        } else if (response.status === 409) {
          // Safety net: the pre-check missed dependents (raced or failed). The
          // 409 payload carries the same dependents shape — reopen the dialog
          // with it instead of dying silently (#317).
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

    const cancelDelete = () => {
      setDeleteConfirmation({ show: false, model: null, dependents: null });
    };

    const closeErrorModal = () => {
      setErrorMessage("");
    };

    const closeSuccessModal = () => {
      setSuccessMessage("");
    };

    // Filter models based on search term
    const filteredModels = models.filter((model) =>
      model.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
      <>
        <div className={`text-gray-200 w-full flex flex-col ${!isLocal ? "h-full" : ""}`}>
          {/* Section header */}
          <div
            className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-700/30 transition-colors flex-shrink-0"
            onClick={() => setOpenSection((prev) => !prev)}
          >
            <div className="flex items-center gap-3">
              {getIconForSection(kind)}
              <span className="font-bold text-lg text-gray-200">{title}</span>
              <TooltipIcon />
              {isLocal && (
                <RefreshCcw
                  className="w-4 h-4 text-gray-400 hover:text-gray-200 cursor-pointer transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                    reloadLocalModels();
                  }}
                />
              )}
            </div>
            <div className="flex items-center gap-2">
              <ChevronDown
                className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${!openSection ? "-rotate-90" : ""}`}
              />
            </div>
          </div>

          {/* Collapsible content */}
          <div
            className={`grid transition-all duration-300 ease-in-out ${!isLocal ? "flex-1 min-h-0" : ""} ${openSection ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"}`}
          >
            <div className="overflow-hidden flex flex-col">
              {/* Search bar for Remote Models only */}
              {hasSearch && !isLocal && openSection && (
                <div className="px-4 py-1 pb-3 flex-shrink-0">
                  <div className="relative rounded-2xl bg-[#1a1a1a]/60 border-[0.2px] border-white/10">
                    <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
                    <input
                      type="text"
                      placeholder={t("landing:sidebar.searchPlaceholder")}
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full bg-transparent rounded-2xl pl-10 pr-4 py-1 text-sm text-white placeholder-gray-400 focus:outline-none border-[0.2px] focus:border-white/10"
                    />
                  </div>
                </div>
              )}

              {/* Models list */}
              <div
                className={`pl-4 pr-4 overflow-y-auto custom-scroll ${!isLocal ? "flex-1 min-h-0" : "max-h-[40vh]"}`}
              >
                {loading ? (
                  <div className="flex items-center gap-2 py-2 text-gray-400">
                    <div className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                    <span className="text-sm">{t("landing:sidebar.loading")}</span>
                  </div>
                ) : filteredModels.length > 0 || models.length > 0 ? (
                  (hasSearch ? filteredModels : models).length > 0 ? (
                    (hasSearch ? filteredModels : models).map((m) =>
                      isLocal ? (
                        <div
                          key={m.id}
                          className="flex items-center justify-between py-1.5 group hover:bg-gray-700/20 rounded px-2 transition-colors"
                        >
                          <span className="flex-1 text-gray-300 text-sm truncate pr-2">
                            {m.name}
                          </span>
                          <button
                            onClick={(e) => handleDeleteClick(e, m)}
                            className="text-red-400 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60 transition-opacity duration-150 hover:text-red-300 p-1 rounded hover:bg-red-900/20"
                            title={t("landing:sidebar.deleteModel")}
                            aria-label={t("landing:sidebar.deleteModel")}
                          >
                            <Trash2 aria-hidden="true" className="w-4 h-4" />
                          </button>
                        </div>
                      ) : m.runnable === false ? (
                        <div
                          key={m.id}
                          className="py-1.5 px-2 text-sm text-gray-600 rounded flex items-center justify-between gap-2 cursor-not-allowed"
                          title={t("landing:sidebar.unavailableTitle")}
                        >
                          <span className="truncate">{m.name}</span>
                          <span className="flex-shrink-0 text-[10px] uppercase tracking-wide text-amber-500/80 border border-amber-500/30 rounded px-1.5 py-0.5">
                            {t("landing:sidebar.unavailableBadge")}
                          </span>
                        </div>
                      ) : (
                        <div
                          key={m.id}
                          className="py-1.5 px-2 text-sm text-gray-400 cursor-pointer hover:text-gray-200 hover:bg-gray-700/20 rounded transition-colors truncate"
                          onClick={() => handleModelClick(m)}
                        >
                          {m.name}
                        </div>
                      )
                    )
                  ) : (
                    <p className="text-gray-500 text-sm italic py-2">
                      {t("landing:sidebar.noneFound")}
                    </p>
                  )
                ) : (
                  <p className="text-gray-500 text-sm italic py-2">
                    {isLocal ? t("landing:sidebar.noneLocal") : t("landing:sidebar.noneRemote")}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Error Modal */}
        <MessageModal
          isOpen={!!errorMessage}
          title={t("common:status.error")}
          message={errorMessage}
          type="error"
          onClose={closeErrorModal}
        />

        {/* Success Modal */}
        <MessageModal
          isOpen={!!successMessage}
          title={t("landing:messages.successTitle")}
          message={successMessage}
          type="success"
          onClose={closeSuccessModal}
        />

        {/* Delete Confirmation Modal */}
        <DeleteModelModal
          isOpen={deleteConfirmation.show}
          model={deleteConfirmation.model}
          dependents={deleteConfirmation.dependents}
          onConfirm={confirmDelete}
          onCancel={cancelDelete}
        />
      </>
    );
  }
);

CollapsibleSection.displayName = "CollapsibleSection";

CollapsibleSection.propTypes = {
  /** Which list the section shows: the installed models or the download catalog. */
  kind: PropTypes.oneOf(["local", "remote"]),
  onLocalModelRefresh: PropTypes.func,
  hasSearch: PropTypes.bool,
};

CollapsibleSection.defaultProps = {
  kind: "local",
  onLocalModelRefresh: null,
  hasSearch: false,
};

export default CollapsibleSection;
