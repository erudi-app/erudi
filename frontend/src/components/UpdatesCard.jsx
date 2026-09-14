import React, { useCallback, useEffect, useRef, useState } from "react";
import { Download } from "lucide-react";
import { useTranslation } from "react-i18next";
import SettingsCard from "./SettingsCard";
import { formatPercent } from "../i18n/format";
import { createLogger } from "../utils/logger";

const log = createLogger("UpdatesCard");

/**
 * The Settings card that gives updates a surface of their own (#571).
 *
 * `UpdateBanner` only exists while an updater event is fresh and the user has
 * not dismissed it, and the install it promises is handed to the native
 * installer at quit. That installer aborts when the app is reopened before
 * the swap finishes ("App Still Running Error") and tells the app nothing, so
 * a user who reopens quickly can be stuck on an old version forever with no
 * control anywhere. This card is that control: check, download, install, on
 * demand.
 *
 * It follows the events rather than assuming what they will be. With
 * automatic updates on, `autoDownload` turns a check straight into a
 * download, so the Download button is offered only in the one state where it
 * is real -- an update found and not yet downloading. On mount it asks main
 * for the phase it may have missed (`updater:get-state`), the way the error
 * screen asks `backend:getInfo`: an update staged hours ago is exactly the
 * case this card exists for.
 *
 * Failures stay on the card as one quiet line, and only for a flow the user
 * started -- a background check that fails is main's business, not a
 * notification. No dialog, no popup.
 */
export default function UpdatesCard() {
  const { t } = useTranslation();
  // The version this build is, not the one being offered.
  const [currentVersion, setCurrentVersion] = useState(null);
  // Whether an updater exists at all: electron-updater is loaded in packaged
  // builds only, so in dev there is nothing to check against.
  const [supported, setSupported] = useState(true);
  // "idle" | "checking" | "up-to-date" | "available" | "downloading" | "downloaded" | "error"
  const [phase, setPhase] = useState("idle");
  const [offeredVersion, setOfferedVersion] = useState(null);
  const [percent, setPercent] = useState(0);
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
        setPhase(state.phase);
        setOfferedVersion(state.version ?? null);
        setPercent(state.percent ?? 0);
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
      switch (payload.event) {
        case "checking-for-update":
          setPhase("checking");
          break;
        case "update-available":
          setPhase("available");
          setOfferedVersion(payload.version ?? null);
          setPercent(0);
          break;
        case "update-not-available":
          setPhase("up-to-date");
          setOfferedVersion(null);
          engaged.current = false;
          break;
        case "download-progress":
          setPhase("downloading");
          setPercent(payload.percent ?? 0);
          break;
        case "update-downloaded":
          setPhase("downloaded");
          setOfferedVersion(payload.version ?? null);
          setPercent(100);
          engaged.current = false;
          break;
        case "error":
          // Only what the user is waiting for. Main logs every one of them.
          if (engaged.current) {
            setPhase("error");
          }
          engaged.current = false;
          break;
        default:
          break;
      }
    };

    return window.updaterAPI?.onUpdaterEvent?.(handlePayload);
  }, []);

  const handleCheck = useCallback(async () => {
    engaged.current = true;
    setPhase("checking");
    const result = await window.updaterAPI?.checkNow?.();
    if (result?.reason === "unavailable") {
      setSupported(false);
      setPhase("idle");
      engaged.current = false;
      return;
    }
    if (result && result.ok === false) {
      // The call itself was refused; no updater event will follow it.
      setPhase("error");
      engaged.current = false;
    }
  }, []);

  const handleDownload = useCallback(async () => {
    engaged.current = true;
    setPercent(0);
    setPhase("downloading");
    const result = await window.updaterAPI?.downloadNow?.();
    if (result && result.ok === false) {
      setPhase("error");
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
        return { text: t("settings:updates.found", { version: offeredVersion ?? "" }) };
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
