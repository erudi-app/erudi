import React, { useCallback, useState } from "react";
import PropTypes from "prop-types";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, Check, Copy, Github, Globe } from "lucide-react";
import { useTranslation } from "react-i18next";
import { apiClient } from "../../services/api/client";
import { describeEngineFailure } from "../../utils/engineNotice";
import { createLogger } from "../../utils/logger";

const log = createLogger("EngineFailureModal");

// Where a report goes. Both open in the system browser: the Electron main
// process routes target="_blank" through shell.openExternal.
export const REPORT_ISSUE_URL = "https://github.com/erudi-app/erudi/issues/new";
export const WEBSITE_URL = "https://erudi-app.github.io/erudi/";
// The processor build of Erudi avoids the graphics card altogether, for someone
// who would rather reinstall than carry a setting.
export const RELEASES_URL = "https://github.com/erudi-app/erudi/releases/latest";

/**
 * The decision dialog for a graphics card Erudi's GPU mode cannot use.
 *
 * One component, two entry points: the startup notice (the pre-flight found the
 * problem before anything failed) and a chat turn that died with an engine
 * code. Both arrive as the same `notice` object and are described by
 * `describeEngineFailure`.
 *
 * Nothing here is automatic. The processor fallback is offered, and applying it
 * writes `inference_backend: "cpu"` then restarts the backend, because the
 * engine is chosen once per boot. "Not now" persists nothing: the condition is
 * real and the dialog is expected to come back.
 */
export default function EngineFailureModal({ notice, onDismiss }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [switchFailed, setSwitchFailed] = useState(false);
  const described = describeEngineFailure(notice);

  const handleCopy = useCallback(() => {
    if (!described?.raw) return;
    navigator.clipboard
      ?.writeText(described.raw)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch((error) => log.error("Failed to copy the engine failure details", error));
  }, [described?.raw]);

  const handleSwitchToCpu = useCallback(async () => {
    setSwitching(true);
    setSwitchFailed(false);
    try {
      await apiClient.put("/user_settings/", { inference_backend: "cpu" });
    } catch (error) {
      // The preference did not persist, so restarting would come back on the
      // GPU and look like the switch silently failed. Say so instead.
      log.error("Failed to persist the processor-mode preference", error);
      setSwitching(false);
      setSwitchFailed(true);
      return;
    }
    // The engine is a process-level singleton read once per boot: only a
    // restart makes the new preference take effect.
    try {
      await window.backendAPI?.restartBackend?.();
    } catch (error) {
      log.error("Failed to restart the backend after switching to processor mode", error);
    }
    setSwitching(false);
    onDismiss();
  }, [onDismiss]);

  const facts = described
    ? [
        [t("errors:engine.modal.gpuLabel"), described.gpuName],
        [t("errors:engine.modal.computeCapabilityLabel"), described.computeCapability],
        [t("errors:engine.modal.driverCudaLabel"), described.driverCudaVersion],
        [t("errors:engine.modal.requiredCudaLabel"), described.requiredCudaVersion],
      ].filter(([, value]) => !!value)
    : [];

  return (
    <AnimatePresence>
      {described && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          role="dialog"
          aria-modal={true}
          aria-label={described.title}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[9999] p-4"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="relative w-full max-w-lg max-h-[90vh] overflow-auto custom-scroll"
          >
            <div
              className={[
                "relative w-full rounded-[26px] overflow-hidden",
                "border border-white/10",
                "bg-[rgba(22,40,36,0.55)] backdrop-blur-[18px] saturate-[1.4]",
                "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)]",
              ].join(" ")}
            >
              <div className="relative z-10 p-6">
                <div className="flex items-start gap-3 mb-4">
                  <div className="w-8 h-8 shrink-0 rounded-lg bg-amber-500/20 flex items-center justify-center">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                  </div>
                  <h2 className="text-xl font-semibold tracking-tight text-[#F2F7F4]">
                    {described.title}
                  </h2>
                </div>

                <p className="text-[13px] leading-relaxed text-[#cfd8d4]">{described.detail}</p>
                <p className="text-[13px] leading-relaxed text-[#9fb0aa] mt-2">{described.hint}</p>

                {facts.length > 0 && (
                  <div className="mt-4 rounded-xl border border-white/10 bg-black/20 p-3">
                    <p className="text-[11px] uppercase tracking-wide text-[#8aa39b] mb-2">
                      {t("errors:engine.modal.facts")}
                    </p>
                    <dl className="space-y-1">
                      {facts.map(([label, value]) => (
                        <div key={label} className="flex justify-between gap-4 text-[12px]">
                          <dt className="text-[#9fb0aa]">{label}</dt>
                          <dd className="text-[#e6efeb] font-mono">{value}</dd>
                        </div>
                      ))}
                    </dl>
                  </div>
                )}

                {described.raw && (
                  <div className="mt-4">
                    <div className="flex items-center justify-between mb-1.5">
                      <p className="text-[11px] uppercase tracking-wide text-[#8aa39b]">
                        {t("errors:engine.modal.traceLabel")}
                      </p>
                      <button
                        type="button"
                        onClick={handleCopy}
                        aria-label={t("errors:engine.modal.copyTrace")}
                        title={t("errors:engine.modal.copyTrace")}
                        className="inline-flex items-center gap-1.5 text-[12px] text-[#9fb0aa] hover:text-[#e6efeb] transition-colors"
                      >
                        {copied ? (
                          <Check className="w-3.5 h-3.5 text-green-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                        <span>
                          {copied ? t("common:actions.copied") : t("common:actions.copy")}
                        </span>
                      </button>
                    </div>
                    <pre
                      readOnly
                      className="text-[11px] leading-relaxed font-mono text-[#8aa39b] bg-black/30 rounded-xl px-3 py-2 max-h-48 overflow-auto whitespace-pre-wrap break-words"
                    >
                      {described.raw}
                    </pre>
                  </div>
                )}

                <div className="mt-4 text-[12px] text-[#9fb0aa]">
                  <p>{t("errors:engine.modal.reportIntro")}</p>
                  <div className="flex flex-wrap gap-3 mt-2">
                    <a
                      href={REPORT_ISSUE_URL}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1.5 text-[#e6efeb] hover:text-white underline underline-offset-2"
                    >
                      <Github className="w-3.5 h-3.5" />
                      {t("errors:engine.modal.reportGithub")}
                    </a>
                    <a
                      href={WEBSITE_URL}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1.5 text-[#e6efeb] hover:text-white underline underline-offset-2"
                    >
                      <Globe className="w-3.5 h-3.5" />
                      {t("errors:engine.modal.reportWebsite")}
                    </a>
                  </div>
                </div>

                {switchFailed && (
                  <p className="mt-4 text-[12px] text-red-300">
                    {t("errors:engine.modal.switchFailed")}
                  </p>
                )}

                <div className="mt-5 flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={handleSwitchToCpu}
                    disabled={switching}
                    className="px-4 py-2 rounded-lg text-[13px] font-medium bg-[#34d399] text-[#02130e] transition-opacity hover:opacity-90 disabled:opacity-60"
                  >
                    {switching
                      ? t("errors:engine.modal.switching")
                      : t("errors:engine.modal.switchToCpu")}
                  </button>
                  <button
                    type="button"
                    onClick={onDismiss}
                    className="px-4 py-2 rounded-lg text-[13px] font-medium border border-white/15 text-[#e6efeb] transition-colors hover:border-white/30"
                  >
                    {t("errors:engine.modal.notNow")}
                  </button>
                </div>
                <p className="mt-2.5 text-[12px] text-[#8aa39b]">
                  {t("errors:engine.modal.switchNote")}
                </p>

                <p className="mt-3 text-[12px] text-[#8aa39b]">
                  {t("errors:engine.modal.cpuInstallerNote")}{" "}
                  <a
                    href={RELEASES_URL}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[#e6efeb] hover:text-white underline underline-offset-2"
                  >
                    {t("errors:engine.modal.cpuInstaller")}
                  </a>
                </p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

EngineFailureModal.propTypes = {
  /** The startup `engine_notice` event, or a chat error event carrying a code. */
  notice: PropTypes.shape({
    code: PropTypes.string,
    gpu_name: PropTypes.string,
    compute_capability: PropTypes.string,
    driver_cuda_version: PropTypes.string,
    required_cuda_version: PropTypes.string,
    raw: PropTypes.string,
  }),
  onDismiss: PropTypes.func.isRequired,
};
