import React, { useCallback, useEffect, useRef, useState } from "react";
import { Download } from "lucide-react";
import { useTranslation } from "react-i18next";
import SettingsCard from "./SettingsCard";
import { formatPercent } from "../i18n/format";
import { updaterStateAfter } from "../utils/updaterPolicy";
import { createLogger } from "../utils/logger";

const log = createLogger("UpdatesCard");

/**
 * The Settings card that gives updates a surface of their own (#571).
 *
 * `UpdateBanner` only exists while an updater event is fresh and the user has
 * not dismissed it, and an install at quit (Install downloaded updates when I
 * quit) is handed to the native installer. That installer aborts when the app is reopened before
 * the swap finishes ("App Still Running Error") and tells the app nothing, so
 * a user who reopens quickly can be stuck on an old version forever with no
 * control anywhere. This card is that control: check, download, install, on
 * demand.
 *
 * It follows the events through the same reducer as main
 * (`updaterStateAfter`): once a version is on disk, the periodic check that
 * finds it again leaves Install now on the card. With
 * Download updates automatically on, `autoDownload` turns a check straight
 * into a download, so the Download button is offered only in the one state where it
 * is real -- an update found and not yet downloading. On mount it asks main
 * for the phase it may have missed (`updater:get-state`), the way the error
 * screen asks `backend:getInfo`: an update staged hours ago is exactly the
 * case this card exists for.
 *
 * Failures stay on the card as one quiet line, and only for a flow the user
 * started -- a background check that fails is main's business, not a
 * notification. No dialog, no popup.
 */
const IDLE = { phase: "idle", version: null, percent: 0, downloadedVersion: null };

export default function UpdatesCard() {
  const { t } = useTranslation();
  // The version this build is, not the one being offered.
  const [currentVersion, setCurrentVersion] = useState(null);
  // Whether an updater exists at all: electron-updater is loaded in packaged
  // builds only, so in dev there is nothing to check against.
  const [supported, setSupported] = useState(true);
  // {phase, version, percent, downloadedVersion}; phases: "idle" | "checking" |
  // "up-to-date" | "available" | "downloading" | "downloaded" | "error".
  const [updater, setUpdater] = useState(IDLE);
  // A download the user started failed: said on the card, which offers it again.
  const [downloadFailed, setDownloadFailed] = useState(false);
  const { phase, version: offeredVersion, percent } = updater;
  // What the card said before a check, to go back to when a background check
  // (not one this user started) fails.
  const beforeCheck = useRef(IDLE);
  useEffect(() => {
    if (phase !== "checking") beforeCheck.current = updater;
  }, [phase, updater]);
  // A flow this user started. An updater error is only ever shown inside one,
  // and nothing renders from it, so it is a ref: flipping it must not redraw.
  const engaged = useRef(false);

  useEffect(() => {
    let cancelled = false;

    const applyState = (state) => {
      if (cancelled || !state) {
        return;
      }
      setSupported(state.available !== false);
      if (state.phase && state.phase !== "idle" && state.phase !== "error") {
        setUpdater({
          phase: state.phase,
          version: state.version ?? null,
          percent: state.percent ?? 0,
          downloadedVersion: state.downloadedVersion ?? null,
        });
      }
    };

    // No bridge at all (a browser, a test) is not a failure: there is simply
    // no updater on this side.
    if (window.updaterAPI?.getState) {
      window.updaterAPI
        .getState()
        .then(applyState)
        .catch((error) => {
          // The card still works -- the events drive it -- so this is a
          // degraded read, not a broken feature.
          log.warn("The app process did not answer the updater state request", error);
        });
    } else {
      setSupported(false);
    }

    window.diagnosticsAPI
      ?.getAppInfo?.()
      .then((info) => {
        if (!cancelled) {
          setCurrentVersion(info?.version ?? null);
        }
      })
      .catch((error) => {
        log.warn("The app process did not answer the version request", error);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const handlePayload = (payload) => {
      if (payload.event === "error") {
        // Only what the user is waiting for is said. Main logs every one.
        const wasEngaged = engaged.current;
        engaged.current = false;
        setUpdater((prev) => {
          const next = updaterStateAfter(prev, payload);
          if (next.phase !== "error" || wasEngaged) return next;
          // Never left on a disabled "Checking..." by a check nobody asked for.
          return prev.phase === "checking" ? beforeCheck.current : prev;
        });
        if (wasEngaged) setDownloadFailed(true);
        return;
      }
      if (payload.event === "update-not-available" || payload.event === "update-downloaded") {
        engaged.current = false;
      }
      setDownloadFailed(false);
      setUpdater((prev) => updaterStateAfter(prev, payload));
    };

    return window.updaterAPI?.onUpdaterEvent?.(handlePayload);
  }, []);

  const handleCheck = useCallback(async () => {
    engaged.current = true;
    setDownloadFailed(false);
    setUpdater((prev) => ({ ...prev, phase: "checking" }));
    const result = await window.updaterAPI?.checkNow?.();
    if (result?.reason === "unavailable") {
      setSupported(false);
      setUpdater((prev) => ({ ...prev, phase: "idle" }));
      engaged.current = false;
      return;
    }
    if (result && result.ok === false) {
      // The call itself was refused; no updater event will follow it.
      setUpdater((prev) => ({ ...prev, phase: "error" }));
      engaged.current = false;
    }
  }, []);

  const handleDownload = useCallback(async () => {
    engaged.current = true;
    setDownloadFailed(false);
    setUpdater((prev) => ({ ...prev, phase: "downloading", percent: 0 }));
    const result = await window.updaterAPI?.downloadNow?.();
    if (result && result.ok === false) {
      // No updater event follows a refused call: the update is still there.
      setUpdater((prev) => ({ ...prev, phase: "available", percent: 0 }));
      setDownloadFailed(true);
      engaged.current = false;
    }
  }, []);

  const handleInstall = useCallback(() => {
    window.updaterAPI?.installNow?.();
  }, []);

  const status = () => {
    if (!supported) {
      return { text: t("settings:updates.unavailable") };
    }
    switch (phase) {
      case "up-to-date":
        return { text: t("settings:updates.upToDate") };
      case "available":
        return downloadFailed
          ? { text: t("settings:updates.failed"), failed: true }
          : { text: t("settings:updates.found", { version: offeredVersion ?? "" }) };
      case "downloading":
        return {
          text: t("settings:updates.downloading", {
            percent: formatPercent(percent, { maximumFractionDigits: 0 }),
          }),
        };
      case "downloaded":
        return { text: t("settings:updates.ready", { version: offeredVersion ?? "" }) };
      case "error":
        return { text: t("settings:updates.failed"), failed: true };
      default:
        return null;
    }
  };

  const buttonClass = [
    "text-[13px] rounded-lg border border-[var(--line)] bg-[var(--canvas)] text-[var(--ink)]",
    "px-3 py-1.5 hover:border-[var(--fit-good)] focus:outline-none focus:border-[var(--fit-good)]",
    "transition-colors whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed",
    "disabled:hover:border-[var(--line)]",
  ].join(" ");

  const action = () => {
    if (!supported) {
      return (
        <button type="button" className={buttonClass} disabled>
          {t("settings:updates.check")}
        </button>
      );
    }
    if (phase === "downloading") {
      return null;
    }
    if (phase === "downloaded") {
      return (
        <button type="button" onClick={handleInstall} className={buttonClass}>
          {t("settings:updates.install")}
        </button>
      );
    }
    if (phase === "available") {
      return (
        <button type="button" onClick={handleDownload} className={buttonClass}>
          {t("settings:updates.download")}
        </button>
      );
    }
    return (
      <button
        type="button"
        onClick={handleCheck}
        className={buttonClass}
        disabled={phase === "checking"}
      >
        {phase === "checking" ? t("settings:updates.checking") : t("settings:updates.check")}
      </button>
    );
  };

  const line = status();

  return (
    <SettingsCard
      icon={<Download className="w-5 h-5 text-[var(--fit-good)]" />}
      title={t("settings:updates.title")}
      description={t("settings:updates.description")}
      note={
        currentVersion
          ? t("settings:updates.currentVersion", { version: currentVersion })
          : undefined
      }
      control={
        <div className="flex flex-col items-end gap-2">
          {action()}
          {line && (
            <p
              className={[
                "text-[12px] leading-relaxed text-right max-w-[14rem]",
                line.failed ? "text-[var(--fit-heavy)]" : "text-[var(--ink-dim)]",
              ].join(" ")}
            >
              {line.text}
            </p>
          )}
        </div>
      }
    />
  );
}
