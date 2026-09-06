import React, { useState } from "react";
import PropTypes from "prop-types";
import {
  Brain,
  MessageSquare,
  Swords,
  BookOpen,
  PanelLeftClose,
  PanelLeftOpen,
  Bug,
  Settings,
} from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useDownloadModal } from "../contexts/DownloadModalContext";
import { DIAGNOSTICS_PATH, SETTINGS_PATH } from "../utils/routes";
import useBugCounter from "../shared/hooks/useBugCounter";

/**
 * Sidebar with icons that highlight based on the current route.
 */
export default function Sidebar({
  disabled = false,
  onToggleSidebar,
  showCollapsible = false,
  collapsed = false,
  showBrainCollapsible = false,
  onToggleBrainSidebar,
  brainCollapsed = false,
}) {
  const { t } = useTranslation();
  const [isHovering, setIsHovering] = useState(false);
  const [isBrainHovering, setIsBrainHovering] = useState(false);
  const { isDownloading } = useDownloadModal();
  const { count: bugCount, label: bugBadgeLabel } = useBugCounter();
  const location = useLocation();
  const isModelsActive = location.pathname === "/erudi/models";
  const isChatActive =
    location.pathname.startsWith("/erudi/chat") ||
    location.pathname.startsWith("/erudi/conversations");
  const isArenaActive = location.pathname === "/erudi/arena";
  const isKnowledgeBaseActive = location.pathname === "/erudi/attach_knowledge_base";
  const isSettingsActive = location.pathname === SETTINGS_PATH;
  const isDiagnosticsActive = location.pathname === DIAGNOSTICS_PATH;

  return (
    <div
      className={`w-14 bg-[#121212] mt-0 flex flex-col items-center transition-opacity duration-200 ${
        disabled ? "opacity-50 pointer-events-none select-none" : ""
      }`}
    >
      {showBrainCollapsible ? (
        <button
          aria-label={t("common:nav.toggleModelsSidebar")}
          onClick={onToggleBrainSidebar}
          onMouseEnter={() => setIsBrainHovering(true)}
          onMouseLeave={() => setIsBrainHovering(false)}
          className={`w-full flex justify-center items-center py-5 border-l-4 ${
            isModelsActive ? "border-green-500" : "border-transparent"
          }`}
        >
          {isBrainHovering ? (
            brainCollapsed ? (
              <PanelLeftOpen
                className={`w-5 h-5 ${isModelsActive ? "text-green-400" : "text-gray-400"}`}
              />
            ) : (
              <PanelLeftClose
                className={`w-5 h-5 ${isModelsActive ? "text-green-400" : "text-gray-400"}`}
              />
            )
          ) : (
            <Brain className={`w-5 h-5 ${isModelsActive ? "text-green-400" : "text-gray-400"}`} />
          )}
        </button>
      ) : (
        <Link
          to="/erudi/models"
          aria-label={t("common:nav.models")}
          className={`w-full flex justify-center items-center py-5 border-l-4 ${
            isModelsActive ? "border-green-500" : "border-transparent"
          }`}
        >
          <Brain
            className={`w-5 h-5 transition-colors duration-200 ${
              isModelsActive ? "text-green-400" : "text-gray-400 hover:text-green-400"
            }`}
          />
        </Link>
      )}

      {showCollapsible ? (
        <button
          aria-label={t("common:nav.toggleChatSidebar")}
          onClick={onToggleSidebar}
          onMouseEnter={() => setIsHovering(true)}
          onMouseLeave={() => setIsHovering(false)}
          className={`w-full flex justify-center items-center py-5 border-l-4 ${
            isChatActive ? "border-green-500" : "border-transparent"
          }`}
        >
          {isHovering ? (
            collapsed ? (
              <PanelLeftOpen
                className={`w-5 h-5 ${isChatActive ? "text-green-400" : "text-gray-400"}`}
              />
            ) : (
              <PanelLeftClose
                className={`w-5 h-5 ${isChatActive ? "text-green-400" : "text-gray-400"}`}
              />
            )
          ) : (
            <MessageSquare
              className={`w-5 h-5 ${isChatActive ? "text-green-400" : "text-gray-400"}`}
            />
          )}
        </button>
      ) : (
        <Link
          to="/erudi/chat"
          aria-label={t("common:nav.chat")}
          className={`w-full flex justify-center items-center py-5 border-l-4 ${
            isChatActive ? "border-green-500" : "border-transparent"
          }`}
        >
          <MessageSquare
            className={`w-5 h-5 transition-colors duration-200 ${
              isChatActive ? "text-green-400" : "text-gray-400 hover:text-green-400"
            }`}
          />
        </Link>
      )}
      <Link
        to="/erudi/arena"
        aria-label={t("common:nav.arena")}
        className={`w-full flex justify-center items-center py-5 border-l-4 ${
          isArenaActive ? "border-green-500" : "border-transparent"
        }`}
      >
        <Swords
          className={`w-5 h-5 transition-colors duration-200 ${
            isArenaActive ? "text-green-400" : "text-gray-400 hover:text-green-400"
          }`}
        />
      </Link>
      <Link
        to="/erudi/attach_knowledge_base"
        aria-label={t("common:nav.knowledgeBase")}
        className={`w-full flex justify-center items-center py-5 border-l-4 ${
          isKnowledgeBaseActive ? "border-green-500" : "border-transparent"
        }`}
      >
        <BookOpen
          className={`w-5 h-5 transition-colors duration-200 ${
            isKnowledgeBaseActive ? "text-green-400" : "text-gray-400 hover:text-green-400"
          }`}
        />
      </Link>

      {/* Settings + Bug Report - Bottom of sidebar */}
      <div className="flex-1" />
      <Link
        to={SETTINGS_PATH}
        aria-label={t("common:nav.settings")}
        className={`w-full flex justify-center items-center py-5 border-l-4 ${
          isSettingsActive ? "border-green-500" : "border-transparent"
        }`}
      >
        <Settings
          className={`w-5 h-5 transition-colors duration-200 ${
            isSettingsActive ? "text-green-400" : "text-gray-400 hover:text-green-400"
          }`}
        />
      </Link>
      {!isDownloading && (
        // Reporting a bug starts on the Diagnostics page: it shows the user
        // what to send, lets them copy it, and takes them to the issue form or
        // the contact page from there. A destination like the others, so it
        // highlights like the others.
        //
        // The small badge (#485) is the whole surface for errors the app
        // recovered from silently: it counts real defects (ERROR/CRITICAL,
        // environmental failures excluded -- see utils/bugCounter.js) recorded
        // since the later of app launch and the last Diagnostics visit, caps
        // its text at "9+", and disappears the moment that page is opened
        // (DiagnosticsPage marks the visit on mount). Nothing pops up, ever:
        // no dialog, no toast, no sound -- this number is it.
        <Link
          to={DIAGNOSTICS_PATH}
          aria-label={
            bugCount > 0
              ? t("common:nav.reportBugWithCount", { count: bugCount })
              : t("common:nav.reportBug")
          }
          className={`w-full flex justify-center items-center py-5 border-l-4 mb-4 ${
            isDiagnosticsActive ? "border-green-500" : "border-transparent"
          }`}
        >
          <span className="relative inline-flex">
            <Bug
              className={`w-5 h-5 transition-colors duration-200 ${
                isDiagnosticsActive ? "text-green-400" : "text-gray-400 hover:text-green-400"
              }`}
            />
            {bugCount > 0 && (
              <span
                data-testid="bug-badge"
                aria-hidden="true"
                className="absolute -top-1.5 -right-2 min-w-[16px] h-4 px-[3px] flex items-center justify-center rounded-full bg-red-500 text-white text-[10px] font-semibold leading-none"
              >
                {bugBadgeLabel}
              </span>
            )}
          </span>
        </Link>
      )}
    </div>
  );
}

Sidebar.propTypes = {
  disabled: PropTypes.bool,
  onToggleSidebar: PropTypes.func,
  isSidebarCollapsed: PropTypes.bool,
};
