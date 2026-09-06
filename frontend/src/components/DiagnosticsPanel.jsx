import React, { useCallback, useEffect, useMemo, useState } from "react";
import PropTypes from "prop-types";
import { CircleCheck, FolderOpen } from "lucide-react";
import { useTranslation } from "react-i18next";

import ReportProblem from "./ReportProblem";
import { apiClient } from "../services/api/client";
import { getSessionErrors } from "../utils/errorCapture";
import { buildPrefill, formatDiagnosticsText, mergeRecentErrors } from "../utils/diagnosticsReport";
import { createLogger } from "../utils/logger";

const log = createLogger("DiagnosticsPanel");

/** Errors shown on screen. The copied text carries the same list. */
const DISPLAY_LIMIT = 200;

/** One label/value row of the environment summary. */
function Row({ label, value }) {
  if (value === null || value === undefined || value === "") return null;
  return (
    <div className="flex items-baseline gap-3 py-1">
      <span className="w-40 shrink-0 text-[12px] text-[var(--ink-faint)]">{label}</span>
      <span className="text-[12px] font-mono text-[var(--ink-dim)] break-all">{value}</span>
    </div>
  );
}

Row.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
};

/**
 * The content of the Diagnostics page: everything a bug report needs, read
 * from this machine and shown to its owner.
 *
 * Three sources are merged: the backend's own log (over loopback), the app log
 * (through the preload bridge) and this window's session error buffer. Each is
 * optional and the panel degrades to whatever it could reach. That is the
 * point: the backend being unreachable is the single most likely reason
 * somebody opens this, so the backend's absence is reported as a fact rather
 * than blanking the screen.
 *
 * When nothing was recorded, the errors area says so and asks nothing: the
 * page is also where people look to check that all is well. The report block
 * stays, for whatever they saw that the logs did not.
 *
 * Nothing here is sent anywhere. The user copies the text and decides where it
 * goes.
 *
 * @returns {JSX.Element} The panel.
 */
export default function DiagnosticsPanel() {
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  const [backend, setBackend] = useState(null);
  const [app, setApp] = useState(null);
  const [appLog, setAppLog] = useState([]);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      // Every source is settled independently: one failure must not take the
      // others with it.
      const [backendResult, appResult, logResult] = await Promise.allSettled([
        apiClient.get("/diagnostics/"),
        window.diagnosticsAPI?.getAppInfo?.() ?? Promise.resolve(null),
        window.diagnosticsAPI?.appLogTail?.(DISPLAY_LIMIT) ?? Promise.resolve([]),
      ]);
      if (cancelled) return;

      if (backendResult.status === "fulfilled") {
        setBackend(backendResult.value);
      } else {
        log.warn("The backend did not answer the diagnostics request", backendResult.reason);
      }
      if (appResult.status === "fulfilled") {
        setApp(appResult.value);
      } else {
        // Half the report comes from the main process. Losing it silently
        // makes this page the next thing to debug, with nothing to go on.
        log.warn("The app process did not answer the environment request", appResult.reason);
      }
      if (logResult.status === "fulfilled" && Array.isArray(logResult.value)) {
        setAppLog(logResult.value);
      } else {
        log.warn(
          "The app log could not be read",
          logResult.status === "rejected" ? logResult.reason : logResult.value
        );
      }
      setLoading(false);
    };

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const entries = useMemo(
    () =>
      mergeRecentErrors({
        backend,
        appLog,
        sessionErrors: getSessionErrors(),
        limit: DISPLAY_LIMIT,
      }),
    [backend, appLog]
  );

  const diagnostics = useMemo(
    () => formatDiagnosticsText({ app, backend, entries }),
    [app, backend, entries]
  );

  const prefill = useMemo(() => buildPrefill({ app, backend }), [app, backend]);

  const openLogFolder = useCallback(() => {
    // Prefer the backend log: it is the one a maintainer asks for first. The
    // main process validates the path and falls back to the app log.
    window.diagnosticsAPI
      ?.revealLog?.(backend?.environment?.backend_log_path ?? null)
      ?.then?.((result) => {
        // Main answers `{success:false}` when it refused the path or could
        // not drive the file manager: the click then does nothing at all, and
        // nothing on screen says why.
        if (result && result.success === false) {
          log.warn("The main process could not reveal the log folder", result);
        }
      })
      ?.catch?.((error) => log.warn("Could not reveal the log folder", error));
  }, [backend]);

  const environment = backend?.environment ?? null;
  const sourceLabel = {
    backend: t("diagnostics:recentErrors.sourceBackend"),
    app: t("diagnostics:recentErrors.sourceApp"),
    session: t("diagnostics:recentErrors.sourceSession"),
  };

  return (
    <section className="relative overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--surface)] rise">
      <div
        className="pointer-events-none absolute -right-24 -top-24 w-72 h-72 rounded-full blur-3xl"
        style={{ background: "radial-gradient(circle, rgba(52,214,165,0.10), transparent 70%)" }}
      />
      <div className="relative p-6 space-y-6">
        {loading && (
          <p className="text-[12px] text-[var(--ink-faint)]">{t("diagnostics:loading")}</p>
        )}

        {!loading && !environment && (
          <div className="rounded-lg border border-[var(--line)] bg-[var(--canvas)] p-3">
            <p className="text-[13px] font-semibold text-[var(--ink)]">
              {t("diagnostics:backendUnavailable.title")}
            </p>
            <p className="text-[12px] text-[var(--ink-dim)] mt-1 leading-relaxed">
              {t("diagnostics:backendUnavailable.body")}
            </p>
          </div>
        )}

        <div>
          <h3 className="text-[13px] font-semibold text-[var(--ink)] mb-2">
            {t("diagnostics:environment.title")}
          </h3>
          <Row label={t("diagnostics:environment.appVersion")} value={app?.version} />
          <Row
            label={t("diagnostics:environment.platform")}
            value={
              environment
                ? [environment.platform, environment.platform_release].filter(Boolean).join(" ")
                : app?.platform
            }
          />
          <Row
            label={t("diagnostics:environment.architecture")}
            value={environment?.architecture ?? app?.arch}
          />
          <Row label={t("diagnostics:environment.engine")} value={environment?.engine} />
          <Row
            label={t("diagnostics:environment.hardware")}
            value={
              environment
                ? [
                    environment.cpu_model,
                    environment.gpu_name,
                    environment.vram_total_gb ? `${environment.vram_total_gb} GB VRAM` : null,
                    environment.compute_capability
                      ? `compute ${environment.compute_capability}`
                      : null,
                  ]
                    .filter(Boolean)
                    .join(" / ") || null
                : null
            }
          />
          <Row
            label={t("diagnostics:environment.model")}
            value={
              environment
                ? (environment.loaded_model ??
                  (environment.loaded_model_id === null
                    ? t("diagnostics:environment.none")
                    : String(environment.loaded_model_id)))
                : null
            }
          />
          <Row label={t("diagnostics:environment.python")} value={environment?.python_version} />
          <Row label={t("diagnostics:environment.database")} value={environment?.db} />
        </div>

        <div>
          <h3 className="text-[13px] font-semibold text-[var(--ink)] mb-2">
            {t("diagnostics:recentErrors.title")}
          </h3>
          {entries.length === 0 ? (
            // Nothing recorded, nothing to do: one line and a check mark, so
            // the page reads as "all is well" rather than as a form to fill.
            // Not before the sources answered: the mark would be a guess.
            !loading && (
              <p className="flex items-center gap-2 text-[12px] text-[var(--ink-dim)]">
                <CircleCheck className="w-3.5 h-3.5 shrink-0 text-[var(--fit-good)]" />
                <span>{t("diagnostics:recentErrors.empty")}</span>
              </p>
            )
          ) : (
            <ul className="max-h-64 overflow-auto custom-scroll rounded-lg border border-[var(--line)] bg-[var(--canvas)] divide-y divide-[var(--line)]">
              {entries.map((entry, index) => (
                <li key={`${entry.timestamp}-${index}`} className="p-2.5">
                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-[var(--ink-faint)]">
                    <span className="font-mono">{entry.timestamp}</span>
                    <span className="font-semibold">{entry.level}</span>
                    <span>{sourceLabel[entry.source]}</span>
                    {entry.requestId && <span className="font-mono">{entry.requestId}</span>}
                    {entry.count > 1 && (
                      <span>{t("diagnostics:recentErrors.repeated", { count: entry.count })}</span>
                    )}
                  </div>
                  <pre className="mt-1 text-[11px] font-mono text-[var(--ink-dim)] whitespace-pre-wrap break-words">
                    {entry.message}
                  </pre>
                </li>
              ))}
            </ul>
          )}
        </div>

        <button
          type="button"
          onClick={openLogFolder}
          className="inline-flex items-center gap-1.5 text-[13px] rounded-lg border border-[var(--line)] bg-[var(--canvas)] text-[var(--ink)] px-3 py-1.5 hover:border-[var(--fit-good)] focus:outline-none focus:border-[var(--fit-good)] transition-colors"
        >
          <FolderOpen className="w-3.5 h-3.5" />
          {t("diagnostics:openLogFolder")}
        </button>

        {/* The lighter rendering: no text preview (the configuration and the
            errors are already listed above), and the copy button only earns
            its place when there is something recorded to copy. */}
        <ReportProblem
          diagnostics={diagnostics}
          prefill={prefill}
          preview={false}
          hasErrors={entries.length > 0}
        />
      </div>
    </section>
  );
}
