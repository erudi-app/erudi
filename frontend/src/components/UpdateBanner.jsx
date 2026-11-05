import React, { useState, useEffect } from "react";
import { Download, RefreshCw, X } from "lucide-react";
import { Trans, useTranslation } from "react-i18next";
import { formatPercent } from "../i18n/format";
import { bannerStateAfter } from "../utils/updaterPolicy";

/**
 * Non-intrusive update notification banner (#563).
 *
 * Phases, from the updater events main forwards (`bannerStateAfter`):
 *   1. available   -- a version was found and nothing downloads on its own
 *                     (Download updates automatically is off): it names the
 *                     version and offers Download;
 *   2. downloading -- progress, from a Download press or an automatic download;
 *   3. ready       -- the update is on disk: Install now.
 * Whether a downloaded update also installs at quit is the "Install
 * downloaded updates when I quit" switch, not the banner's business.
 *
 * Dismissing hides the banner for that version until the next launch: the
 * periodic check finds it again and must not bring it back. The Update Erudi
 * card in Settings still offers it.
 */
export default function UpdateBanner() {
  const { t } = useTranslation();
  const [state, setState] = useState(null);
  const [dismissedVersion, setDismissedVersion] = useState(null);

  useEffect(() => {
    return window.updaterAPI?.onUpdaterEvent?.((payload) => {
      setState((prev) => bannerStateAfter(prev, payload));
    });
  }, []);

  if (!state || state.version === dismissedVersion) {
    return null;
  }

  const handleDownload = async () => {
    setState((prev) => ({ phase: "downloading", version: prev?.version ?? "", percent: 0 }));
    const result = await window.updaterAPI?.downloadNow?.();
    if (result && result.ok === false) {
      // No updater event follows a refused call: offer the button again.
      setState((prev) =>
        prev?.phase === "downloading" ? { phase: "available", version: prev.version } : prev
      );
    }
  };

  const handleInstall = () => {
    window.updaterAPI?.installNow?.();
  };

  const actionClass =
    "shrink-0 rounded-lg border border-[#00B574]/40 px-2.5 py-1 text-[12px] text-[#00B574] hover:text-white hover:border-white/40 transition-colors";

  return (
    <div
      className={[
        "fixed bottom-4 right-4 z-50 flex items-center gap-3 px-4 py-3 rounded-xl",
        "bg-[rgba(22,40,36,0.92)] backdrop-blur-[12px] saturate-[1.3]",
        "border border-[#00B574]/30",
        "shadow-[0_4px_20px_rgba(0,0,0,0.4)]",
        "text-sm text-white max-w-sm",
        "animate-in slide-in-from-bottom-2 duration-300",
      ].join(" ")}
    >
      <div className="shrink-0">
        {state.phase === "ready" ? (
          <RefreshCw className="w-4 h-4 text-[#00B574]" />
        ) : (
          <Download className="w-4 h-4 text-[#00B574]" />
        )}
      </div>

      <div className="flex-1 min-w-0">
        {state.phase === "available" && (
          <p className="text-gray-200">
            <Trans
              i18nKey="common:update.available"
              values={{ version: state.version }}
              components={{ v: <span className="text-[#00B574] font-semibold" /> }}
            />
          </p>
        )}

        {state.phase === "downloading" && (
          <>
            <p className="text-gray-200 mb-1">
              <Trans
                i18nKey="common:update.downloading"
                values={{ version: state.version, percent: formatPercent(state.percent) }}
                components={{
                  v: <span className="text-[#00B574] font-semibold" />,
                  pct: <span className="text-gray-400" />,
                }}
              />
            </p>
            <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-[#00B574] rounded-full transition-all duration-300"
                style={{ width: `${state.percent}%` }}
              />
            </div>
          </>
        )}

        {state.phase === "ready" && (
          <p className="text-gray-200">
            <Trans
              i18nKey="common:update.ready"
              values={{ version: state.version }}
              components={{ v: <span className="text-[#00B574] font-semibold" /> }}
            />
          </p>
        )}
      </div>

      {state.phase === "available" && (
        <button type="button" onClick={handleDownload} className={actionClass}>
          {t("common:update.download")}
        </button>
      )}

      {state.phase === "ready" && (
        <button type="button" onClick={handleInstall} className={actionClass}>
          {t("common:update.install")}
        </button>
      )}

      {/* Dismiss -- on "available" and "ready"; a download in progress stays. */}
      {state.phase !== "downloading" && (
        <button
          type="button"
          onClick={() => setDismissedVersion(state.version)}
          className="shrink-0 text-gray-500 hover:text-gray-300 transition-colors"
          aria-label={t("common:update.dismiss")}
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}
