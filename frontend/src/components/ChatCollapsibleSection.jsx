import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";
import {
  ChevronDown,
  ChevronRight,
  Download,
  RefreshCcw,
  Plus,
  Edit3,
  Search,
  X,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import ErrorModal from "./modals/ErrorModal";
import { API_BASE_URL } from "../config/api.js";
import { tracedFetch } from "../services/api/client";
import { createLogger } from "../utils/logger";
import { conversationPath } from "../utils/routes";
import { exportConversation } from "../utils/conversationExport";
import { abortTurns } from "../utils/conversationTurns";
const log = createLogger("ChatCollapsibleSection");
// Route of the new-chat screen (mirrors the Sidebar link).
const CHAT_PATH = "/erudi/chat";
// The name the backend stores for a conversation with no title (a new one, a
// greeting, a title that did not come out); shown translated, so the stored
// name stays right when the interface language changes.
const DEFAULT_CONVERSATION_NAME = "New Conversation";
// How long the search field waits for a pause in the typing before it asks the
// backend (#429).
export const SEARCH_DEBOUNCE_MS = 300;

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
  onRefresh: PropTypes.func,
  disabled: PropTypes.bool,
};

ChatCollapsibleSection.defaultProps = {
  items: [],
  selectedId: null,
  onSelect: null,
  onRename: null,
  onDelete: null,
  onRefresh: null,
  disabled: false,
};

// The conversations whose DELETE is on its way (shared by every history list).
const deletesInFlight = new Set();

export default function ChatCollapsibleSection({
  title,
  items = [],
  selectedId,
  onSelect,
  onRename,
  onDelete,
  onRefresh,
  disabled = false,
}) {
  const { t } = useTranslation();
  const displayName = (name) =>
    name === DEFAULT_CONVERSATION_NAME ? t("chat:conversation.untitled") : name;
  const [open, setOpen] = useState(true);
  const [editingId, setEditingId] = useState(null);
  const [tempName, setTempName] = useState("");
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  // The search (#429). `outcome` is what the backend answered for the text it
  // was asked about -- `{ text, rows }` or `{ text, failed }` -- and counts only
  // while that text is still what is typed: the results replace the list, and
  // an emptied field brings the list back.
  const [query, setQuery] = useState("");
  const [outcome, setOutcome] = useState(null);
  const typed = query.trim();
  const current = outcome && outcome.text === typed ? outcome : null;
  const results = current?.rows ?? null;
  const searchFailed = !!current?.failed;

  useEffect(() => {
    if (!typed) {
      return undefined;
    }
    // A newer query (or an unmount) retires this one, so a slow answer never
    // overwrites the results of what is typed now.
    let live = true;
    const timer = setTimeout(async () => {
      try {
        const res = await tracedFetch(
          `${API_BASE_URL}/conversations/search?q=${encodeURIComponent(typed)}`
        );
        if (!res.ok) {
          throw new Error(res.status);
        }
        const rows = await res.json();
        if (live) {
          setOutcome({ text: typed, rows });
        }
      } catch (err) {
        if (live) {
          log.error("Conversation search failed", err);
          setOutcome({ text: typed, failed: true });
        }
      }
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      live = false;
      clearTimeout(timer);
    };
    // `items` is the history the parent holds: when it changes under a search
    // (a rename, a new conversation, a delete) the search is asked again, so the
    // results never keep a title or a row the history no longer has.
  }, [typed, items]);

  const changeResults = (change) =>
    setOutcome((prev) => (prev?.rows ? { ...prev, rows: change(prev.rows) } : prev));

  const shown = results ?? items;

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
      changeResults((rows) => rows.map((c) => (c.id === id ? { ...c, name } : c)));
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
  //
  // The answer the conversation is streaming, if any, is aborted first: the
  // backend deletes a conversation only once its turn has ended, and the abort
  // ends it now rather than at the end of the generation. A second delete of
  // the same conversation while the first is on its way sends nothing.
  const deleteConversation = async (id) => {
    if (deletesInFlight.has(String(id))) return;
    deletesInFlight.add(String(id));
    abortTurns(id);
    try {
      const res = await tracedFetch(`${API_BASE_URL}/conversations/${id}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
      });

      if (!res.ok) {
        throw new Error("Delete failed");
      }

      changeResults((rows) => rows.filter((c) => c.id !== id));
      onDelete?.(id);
    } catch (err) {
      log.error(err);
      alert(t("chat:history.deleteFailed", { error: err.message }));
    } finally {
      deletesInFlight.delete(String(id));
      setEditingId(null);
    }
  };

  // The save dialog and the file belong to the main process; a cancelled dialog
  // is not a failure.
  const exportRow = async (id) => {
    try {
      await exportConversation(id, { t });
    } catch (err) {
      log.error(err);
      setErrorMessage(t("chat:export.failed", { error: err.message }));
    }
  };

  // A row action: a button for the mouse and for the keyboard (Tab reaches it,
  // Enter and Space press it), never a click on the row behind it.
  const rowAction = (label, action) => ({
    role: "button",
    tabIndex: 0,
    "aria-label": label,
    onClick: (e) => {
      e.stopPropagation();
      action();
    },
    onKeyDown: (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        e.stopPropagation();
        action();
      }
    },
  });

  const renderItems = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center py-4">
          <div className="w-5 h-5 border-2 border-gray-400 border-t-emerald-400 rounded-full animate-spin"></div>
        </div>
      );
    }

    if (shown.length > 0) {
      return shown.map((conv) => {
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
                placeholder={displayName(conv.name)}
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
              <span>{displayName(conv.name)}</span>
            )}

            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex gap-2 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100">
              <Download
                {...rowAction(t("chat:export.action"), () => exportRow(conv.id))}
                className="w-4 h-4 text-gray-400 hover:text-gray-200 cursor-pointer"
              />
              <Edit3
                {...rowAction(t("chat:history.renameConversation"), () => {
                  setEditingId(conv.id);
                  setTempName(conv.name === DEFAULT_CONVERSATION_NAME ? "" : conv.name);
                })}
                className="w-4 h-4 text-gray-400 hover:text-gray-200 cursor-pointer"
              />
              <X
                {...rowAction(t("chat:history.deleteConversation"), () => {
                  setPendingDeleteId(conv.id);
                  setShowDeleteConfirm(true);
                })}
                className="w-4 h-4 text-red-400 hover:text-red-300 cursor-pointer"
              />
            </div>
          </div>
        );
      });
    }

    return (
      <p className="italic">
        {results !== null ? t("chat:history.noResults") : t("chat:history.empty")}
      </p>
    );
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

      {open && (
        <div className="px-4 pb-2 flex-shrink-0">
          <div className="relative">
            <Search
              aria-hidden="true"
              className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none"
            />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  setQuery("");
                }
              }}
              maxLength={200}
              placeholder={t("chat:history.searchPlaceholder")}
              aria-label={t("chat:history.searchLabel")}
              className="w-full rounded-md bg-gray-700/40 py-1.5 pl-9 pr-8 text-sm text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-emerald-400 [&::-webkit-search-cancel-button]:hidden"
            />
            {query && (
              <X
                role="button"
                aria-label={t("chat:history.clearSearch")}
                onClick={() => setQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 hover:text-gray-200 cursor-pointer"
              />
            )}
          </div>
        </div>
      )}

      <div
        className={`grid transition-all duration-300 ease-in-out flex-1 min-h-0 ${open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"} `}
      >
        <div className="px-4 py-2 overflow-y-auto overflow-x-visible custom-scroll">
          {searchFailed && (
            <p role="alert" className="mb-2 italic text-red-300">
              {t("chat:history.searchFailed")}
            </p>
          )}
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
