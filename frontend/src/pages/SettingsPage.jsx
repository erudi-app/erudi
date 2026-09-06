import React, { useEffect } from "react";
import PropTypes from "prop-types";
import { Cpu, Globe, Languages, RefreshCw, ShieldCheck } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useLocation } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import ToggleSwitch from "../components/ToggleSwitch";
import DiagnosticsPanel from "../components/DiagnosticsPanel";
import { useUserSettings } from "../shared/hooks/api";
import { setAppLanguage } from "../i18n";
import { LANGUAGE_NAMES, SUPPORTED_LANGUAGES } from "../i18n/languages";
import { notifyAutoUpdatePreference } from "../utils/autoUpdate";
import { createLogger } from "../utils/logger";

const log = createLogger("SettingsPage");

// The privacy page lives in the documentation site so every claim on it can
// point at the code that backs it. The window-open handler in main.js routes
// target="_blank" links to the system browser.
export const PRIVACY_PAGE_URL = "https://erudi-app.github.io/erudi/privacy/";

/**
 * One settings card: icon, title, description, optional fine-print note and
 * the control on the right. Shared by every section so they look identical.
 */
function SettingsCard({ icon, title, description, note, control }) {
  return (
    <section className="relative overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--surface)] rise">
      <div
        className="pointer-events-none absolute -right-24 -top-24 w-72 h-72 rounded-full blur-3xl"
        style={{
          background: "radial-gradient(circle, rgba(52,214,165,0.10), transparent 70%)",
        }}
      />
      <div className="relative p-6">
        <div className="flex items-start justify-between gap-6">
          <div className="flex items-start gap-3.5">
            <div className="mt-0.5 rounded-xl border border-[var(--line)] bg-[var(--surface-2)] p-2.5">
              {icon}
            </div>
            <div>
              <h2 className="text-[15px] font-semibold text-[var(--ink)] tracking-tight">
                {title}
              </h2>
              <p className="text-[13px] text-[var(--ink-dim)] mt-1.5 max-w-md leading-relaxed">
                {description}
              </p>
              {note && (
                <p className="text-[12px] text-[var(--ink-faint)] mt-2.5 leading-relaxed">{note}</p>
              )}
            </div>
          </div>
          <div className="pt-1">{control}</div>
        </div>
      </div>
    </section>
  );
}

SettingsCard.propTypes = {
  icon: PropTypes.node.isRequired,
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  note: PropTypes.string,
  control: PropTypes.node.isRequired,
};

/**
 * App-wide settings page (gear icon in the sidebar rail).
 *
 * Sections: the global Web Search default (#310), automatic updates, the
 * inference engine and the application language (#385). Enabling web search
 * lets tool-capable models
 * search the web; the searched query is sent to external search engines, so it
 * ships OFF by default and new conversations inherit whatever the user picks
 * here (each conversation then owns its own toggle). Automatic updates ship ON
 * -- refusing them stops the update traffic entirely, and the choice is handed
 * to the Electron main process, which owns electron-updater. The inference
 * engine defaults to Automatic (the hardware decides) and can be pinned to the
 * processor for a machine whose graphics card Erudi cannot drive; the backend
 * reads it once per boot, so changing it restarts the engine. The language
 * applies immediately through i18next and is persisted with the other settings.
 *
 * The page also carries the Diagnostics panel, the destination of the bug
 * button in the sidebar rail. It shows what a bug report needs about this
 * machine and hands it to the user to copy; it sends nothing anywhere.
 */
export default function SettingsPage() {
  const { t, i18n } = useTranslation();
  const { hash } = useLocation();
  const { settings, loading, updateSettings } = useUserSettings();

  // The sidebar's bug button links to /erudi/settings#diagnostics. Under
  // HashRouter the browser cannot follow that fragment on its own -- the whole
  // route already lives in the hash -- so the scroll happens here.
  useEffect(() => {
    if (hash !== "#diagnostics") return;
    document.getElementById("diagnostics")?.scrollIntoView({ behavior: "smooth" });
  }, [hash]);
  const webSearchEnabled = settings?.web_search_enabled ?? false;
  const autoUpdateEnabled = settings?.auto_update_enabled ?? true;
  const inferenceBackend = settings?.inference_backend ?? "auto";

  const handleWebSearchToggle = async (next) => {
    try {
      await updateSettings({ web_search_enabled: next });
    } catch (error) {
      log.error("Failed to update the web search setting", error);
    }
  };

  const handleAutoUpdateToggle = async (next) => {
    try {
      await updateSettings({ auto_update_enabled: next });
      // Main holds electron-updater and has no database: without this it would
      // keep checking after the user said no.
      notifyAutoUpdatePreference(next);
    } catch (error) {
      log.error("Failed to update the automatic update setting", error);
    }
  };

  const handleInferenceBackendChange = async (event) => {
    const next = event.target.value;
    try {
      await updateSettings({ inference_backend: next });
      // The engine is chosen once per boot, so the new preference only takes
      // effect after the backend comes back up.
      await window.backendAPI?.restartBackend?.();
    } catch (error) {
      log.error("Failed to update the inference engine setting", error);
    }
  };

  const handleLanguageChange = async (event) => {
    const next = event.target.value;
    // Switch the UI first so the change is instant; persistence follows.
    await setAppLanguage(next);
    try {
      await updateSettings({ language: next });
    } catch (error) {
      log.error("Failed to persist the language setting", error);
    }
  };

  return (
    <div className="flex h-screen bg-[#071b18]">
      <Sidebar />
      <main className="flex-1 bg-[var(--canvas)] relative custom-scroll overflow-auto">
        <div className="mx-auto max-w-3xl px-8 py-10 space-y-9">
          <header className="rise">
            <span className="eyebrow">{t("settings:eyebrow")}</span>
            <h1 className="text-2xl font-semibold text-[var(--ink)] tracking-tight mt-1.5">
              {t("settings:title")}
            </h1>
            <p className="text-[13px] text-[var(--ink-dim)] mt-1.5">{t("settings:subtitle")}</p>
          </header>

          <SettingsCard
            icon={<Globe className="w-5 h-5 text-[var(--fit-good)]" />}
            title={t("settings:webSearch.title")}
            description={t("settings:webSearch.description")}
            note={t("settings:webSearch.note")}
            control={
              <ToggleSwitch
                checked={webSearchEnabled}
                onChange={handleWebSearchToggle}
                label={t("settings:webSearch.toggleLabel")}
                disabled={loading}
              />
            }
          />

          <SettingsCard
            icon={<RefreshCw className="w-5 h-5 text-[var(--fit-good)]" />}
            title={t("settings:autoUpdate.title")}
            description={t("settings:autoUpdate.description")}
            note={t("settings:autoUpdate.note")}
            control={
              <ToggleSwitch
                checked={autoUpdateEnabled}
                onChange={handleAutoUpdateToggle}
                label={t("settings:autoUpdate.toggleLabel")}
                disabled={loading}
              />
            }
          />

          <SettingsCard
            icon={<Cpu className="w-5 h-5 text-[var(--fit-good)]" />}
            title={t("settings:inferenceBackend.title")}
            description={t("settings:inferenceBackend.description")}
            note={t("settings:inferenceBackend.note")}
            control={
              <select
                aria-label={t("settings:inferenceBackend.selectLabel")}
                value={inferenceBackend}
                onChange={handleInferenceBackendChange}
                disabled={loading}
                className="text-[13px] rounded-lg border border-[var(--line)] bg-[var(--canvas)] text-[var(--ink)] px-3 py-1.5 focus:outline-none focus:border-[var(--fit-good)] transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <option value="auto">{t("settings:inferenceBackend.auto")}</option>
                <option value="cpu">{t("settings:inferenceBackend.cpu")}</option>
              </select>
            }
          />

          <SettingsCard
            icon={<Languages className="w-5 h-5 text-[var(--fit-good)]" />}
            title={t("settings:language.title")}
            description={t("settings:language.description")}
            note={t("settings:language.note")}
            control={
              <select
                aria-label={t("settings:language.selectLabel")}
                value={i18n.language}
                onChange={handleLanguageChange}
                disabled={loading}
                className="text-[13px] rounded-lg border border-[var(--line)] bg-[var(--canvas)] text-[var(--ink)] px-3 py-1.5 focus:outline-none focus:border-[var(--fit-good)] transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {SUPPORTED_LANGUAGES.map((code) => (
                  <option key={code} value={code} lang={code}>
                    {LANGUAGE_NAMES[code]}
                  </option>
                ))}
              </select>
            }
          />

          <SettingsCard
            icon={<ShieldCheck className="w-5 h-5 text-[var(--fit-good)]" />}
            title={t("settings:privacy.title")}
            description={t("settings:privacy.description")}
            note={t("settings:privacy.note")}
            control={
              <a
                href={PRIVACY_PAGE_URL}
                target="_blank"
                rel="noreferrer"
                className="inline-block text-[13px] rounded-lg border border-[var(--line)] bg-[var(--canvas)] text-[var(--ink)] px-3 py-1.5 hover:border-[var(--fit-good)] focus:outline-none focus:border-[var(--fit-good)] transition-colors whitespace-nowrap"
              >
                {t("settings:privacy.linkLabel")}
              </a>
            }
          />

          {/* The bug button in the sidebar rail links here. It owns its own
              data loading, its own loading and error states and the shared
              report block, so it is a section rather than a SettingsCard. */}
          <DiagnosticsPanel />
        </div>
      </main>
    </div>
  );
}
