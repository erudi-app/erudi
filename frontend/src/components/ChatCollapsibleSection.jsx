import React, { useState } from "react";
import PropTypes from "prop-types";
import { ChevronDown, ChevronRight, RefreshCcw, Plus, Edit3, X, Download } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import ErrorModal from "./modals/ErrorModal";
import { API_BASE_URL } from "../config/api.js";
import { tracedFetch } from "../services/api/client";
import { createLogger } from "../utils/logger";
import { conversationPath } from "../utils/routes";
import { formatConversationToMarkdown } from "../utils/markdownExport";
const log = createLogger("ChatCollapsibleSection");
// Route of the new-chat screen (mirrors the Sidebar link).
const CHAT_PATH = "/erudi/chat";

ChatCollapsibleSection.propTypes = {
  title: PropTypes.string.isRequired,
  items: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      title: PropTypes.string.isRequired,
    })
  ),
  selectedId: PropTypes.string,
  onSelect: PropTypes.func,
  onRename: PropTypes.func,
  onDelete: PropTypes.func,
  onExport: PropTypes.func,
  onRefresh: PropTypes.func,
  disabled: PropTypes.bool,
};

ChatCollapsibleSection.defaultProps = {
  items: [],
  selectedId: null,
  onSelect: null,
  onRename: null,
  onDelete: null,
  onExport: null,
  onRefresh: null,
  disabled: false,
};

export default function ChatCollapsibleSection({
  title,
  items = [],
  selectedId,
  onSelect,
  onRename,
  onDelete,
  onExport,
  onRefresh,
  disabled = false,
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(true);
  const [editingId, setEditingId] = useState(null);
  const [tempName, setTempName] = useState("");
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  const renameConversation = async (id, name) => {
    try {
      const res = await tracedFetch(`${API_BASE_URL}/conversations/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });

      if (!res.ok) {
        throw new Error(res.status);
      }
      onRename?.(id, name);
    } catch (err) {
      log.error(err);
      alert(t("chat:history.renameFailed", { error: err.message }));
    } finally {
      setEditingId(null);
    }
  };

  // Ownership contract (#228): THIS child owns the DELETE mutation. It issues the
  // single DELETE request, then calls onDelete for post-delete UI only. Parents'
  // onDelete callbacks must NOT repeat the DELETE (state/navigation only), or the
  // request fires twice ~10ms apart with distinct request ids.
  const deleteConversation = async (id) => {
    try {
      const res = await tracedFetch(`${API_BASE_URL}/conversations/${id}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
      });

      if (!res.ok) {
        throw new Error("Delete failed");
      }

      onDelete?.(id);
    } catch (err) {
      log.error(err);
      alert(t("chat:history.deleteFailed", { error: err.message }));
    } finally {
      setEditingId(null);
    }
  };

  const exportConversation = async (id) => {
    try {
      const res = await tracedFetch(`${API_BASE_URL}/conversations/${id}`);
      if (!res.ok) {
        throw new Error(res.status);
      }
      const conv = await res.json();
      const mdContent = formatConversationToMarkdown(conv);
      
      const blob = new Blob([mdContent], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const safeName = (conv.name || "conversation").replace(/[^a-z0-9]/gi, '_').toLowerCase();
      a.download = `${safeName}_${id}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      onExport?.(id);
    } catch (err) {
      log.error(err);
      alert(t("chat:history.exportFailed", { error: err.message, defaultValue: "Export failed: {{error}}" }));
    }
  };

  const renderItems = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center py-4">
          <div className="w-5 h-5 border-2 border-gray-400 border-t-emerald-400 rounded-full animate-spin"></div>
        </div>
      );
    }

    if (items.length > 0) {
      return items.map((conv) => {
        const isSelected = selectedId === conv.id;
        const isEditing = editingId === conv.id;

        return (
          <div
            key={conv.id}
            onClick={() => {
              if (!isEditing) {
                onSelect?.(conv.id);
                navigate(conversationPath(conv.id));
              }
            }}
            className={`relative group py-2 px-4 rounded-md cursor-pointer transition-all duration-150 ${
              isSelected
                ? "bg-emerald-500/50 text-white"
                : "hover:bg-gray-700 hover:text-white text-gray-300"
            }`}
          >
            {isEditing ? (
              <input
                value={tempName}
                onChange={(e) => setTempName(e.target.value)}
                onKeyDown={async (e) => {
                  if (e.key === "Enter" && tempName.trim()) {
                    await renameConversation(conv.id, tempName.trim());
                    e.stopPropagation();
                  }
                  if (e.key === "Escape") {
                    setEditingId(null);
                  }
                }}
                onBlur={() => setEditingId(null)}
                autoFocus
                className="w-full bg-transparent focus:outline-none focus:ring-0 focus:shadow-none focus:border-transparent border-b border-emerald-400"
              />
            ) : (
              <span>{conv.name}</span>
            )}

            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex gap-2 opacity-0 group-hover:opacity-100">
              <Download
                role="button"
                aria-label={t("chat:history.exportConversation", { defaultValue: "Export conversation" })}
                onClick={(e) => {
                  e.stopPropagation();
                  exportConversation(conv.id);
                }}
                className="w-4 h-4 text-gray-400 hover:text-gray-200 cursor-pointer"
              />
              <Edit3
                role="button"
                aria-label={t("chat:history.renameConversation")}
                onClick={(e) => {
                  e.stopPropagation();
                  setEditingId(conv.id);
                  setTempName(conv.name);
                }}
                className="w-4 h-4 text-gray-400 hover:text-gray-200 cursor-pointer"
              />
              <X
                role="button"
                aria-label={t("chat:history.deleteConversation")}
                onClick={(e) => {
                  e.stopPropagation();
                  setPendingDeleteId(conv.id);
                  setShowDeleteConfirm(true);
                }}
                className="w-4 h-4 text-red-400 hover:text-red-300 cursor-pointer"
              />
            </div>
          </div>
        );
      });
    }

    return <p className="italic">{t("chat:history.empty")}</p>;
  };

  const closeErrorModal = () => {
    setErrorMessage("");
  };

  const handleRefresh = async (e) => {
    e.stopPropagation();
    setLoading(true);
    await new Promise((resolve) => setTimeout(resolve, 300));
    try {
      await onRefresh?.();
    } catch (err) {
      log.error("Failed to refresh conversations:", err);
      setErrorMessage(
        t("chat:errors.refreshConversations", {
          error: err.message || t("chat:errors.network"),
        })
      );
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = (e) => {
    e.stopPropagation();
    navigate(CHAT_PATH);
  };

  return (
    <div
      className={`text-gray-200 h-full flex flex-col ${disabled ? "pointer-events-none opacity-50 select-none" : ""}`}
    >
      <div
        className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-700/30 flex-shrink-0"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-2">
          {open ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <span className="font-semibold">{title}</span>
        </div>

        <div className="flex gap-3">
          <RefreshCcw
            role="button"
            aria-label={t("chat:history.refreshConversations")}
            className="w-6 h-6 hover:opacity-70 hover:bg-gray-600/30 rounded-full p-1 -m-1 cursor-pointer"
            onClick={handleRefresh}
          />
          <Plus
            role="button"
            aria-label={t("chat:history.newChat")}
            className="w-6 h-6 hover:opacity-70 hover:bg-gray-600/30 rounded-full p-1 -m-1 cursor-pointer"
            onClick={handleNewChat}
          />
        </div>
      </div>

      <div
        className={`grid transition-all duration-300 ease-in-out flex-1 min-h-0 ${open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"} `}
      >
        <div className="px-4 py-2 overflow-y-auto overflow-x-visible custom-scroll">
          {renderItems()}
        </div>
      </div>

      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-[#272727] text-white rounded-lg p-6 w-full max-w-sm shadow-lg">
            <h2 className="text-lg font-semibold mb-4">{t("chat:history.deleteConfirmTitle")}</h2>
            <p className="text-sm mb-6">{t("chat:history.deleteConfirmBody")}</p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 text-sm bg-gray-600 hover:bg-gray-700 rounded"
              >
                {t("common:actions.cancel")}
              </button>
              <button
                onClick={() => {
                  deleteConversation(pendingDeleteId);
                  setShowDeleteConfirm(false);
                }}
                className="px-4 py-2 text-sm bg-red-600 hover:bg-red-700 rounded"
              >
                {t("common:actions.delete")}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error Popup */}
      <ErrorModal errorMessage={errorMessage} onClose={closeErrorModal} />
    </div>
  );
}
