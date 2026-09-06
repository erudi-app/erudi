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
  const location = useLocation();
  const isModelsActive = location.pathname === "/erudi/models";
  const isChatActive =
    location.pathname.startsWith("/erudi/chat") ||
    location.pathname.startsWith("/erudi/conversations");
  const isArenaActive = location.pathname === "/erudi/arena";
  const isKnowledgeBaseActive = location.pathname === "/erudi/attach_knowledge_base";
  const isSettingsActive = location.pathname === SETTINGS_PATH;

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
        // Reporting a bug starts with the Diagnostics panel in Settings: it
        // shows the user what to send, lets them copy it, and takes them to the
        // issue form or the contact page from there.
        <Link
          to={DIAGNOSTICS_PATH}
          aria-label={t("common:nav.reportBug")}
          className="w-full flex justify-center items-center py-5 border-l-4 border-transparent mb-4"
        >
          <Bug className="w-5 h-5 transition-colors duration-200 text-gray-400 hover:text-red-400" />
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
