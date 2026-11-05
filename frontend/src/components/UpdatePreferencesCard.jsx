import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import SettingsCard from "./SettingsCard";
import ToggleSwitch from "./ToggleSwitch";
import { notifyUpdatePreferences, updatePreferencesFrom } from "../utils/autoUpdate";
import { installMayStillHappen, resolveUpdatePreferences } from "../utils/updaterPolicy";
import { createLogger } from "../utils/logger";

const log = createLogger("UpdatePreferencesCard");

// The three switches, in the order they are shown, with their settings field.
const SWITCHES = [
  { key: "check", field: "auto_update_enabled" },
  { key: "download", field: "auto_download_updates" },
  { key: "install", field: "auto_install_updates" },
];

/**
 * The Settings "Updates" card (#563): what the app does on its own when a new
 * version is published.
 *
 * - Check for updates (on by default): asks the update feed at launch and
 *   every 4 hours, and shows a banner. Notify only.
 * - Download updates automatically (off by default).
 * - Install downloaded updates when I quit (off by default).
 * The two opt-ins act only while the check is on, so they are disabled
 * while it is off. Each change is persisted, one at a time (every switch is
 * held while a save runs), then the full snapshot the backend answered is
 * handed to the main process, which owns
 * electron-updater (`updaterPolicy.js` decides what it does with it). A
 * change that did not persist is not handed over.
 *
 * On macOS an update downloaded while install was on is handed to the system
 * installer, which installs it at quit whatever the switches say from then
 * on: main reports that hand-off, and the card says so when install is
 * (effectively) off. The manual Update Erudi card sits below and works
 * whatever this one says.
 */
export default function UpdatePreferencesCard({ settings, loading, updateSettings }) {
  const { t } = useTranslation();
  const [handedOff, setHandedOff] = useState(false);
  // One save at a time: a second toggle could otherwise be answered first and
  // hand main a snapshot the database no longer holds.
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    window.updaterAPI
      ?.getState?.()
      .then((state) => {
        if (!cancelled && state?.handedOff) setHandedOff(true);
      })
      .catch((error) => {
        // The events still report a later hand-off; only the catch-up is lost.
        log.warn("The app process did not answer the updater state request", error);
      });
    const unsubscribe = window.updaterAPI?.onUpdaterEvent?.((payload) => {
      if (
        payload.event === "install-handed-off" ||
        (payload.event === "update-downloaded" && payload.handedOff)
      ) {
        setHandedOff(true);
      }
    });
    return () => {
      cancelled = true;
      if (typeof unsubscribe === "function") unsubscribe();
    };
  }, []);

  const stored = {
    check: settings?.auto_update_enabled ?? true,
    download: settings?.auto_download_updates ?? false,
    install: settings?.auto_install_updates ?? false,
  };

  const handleToggle = async (field, next) => {
    if (saving) return;
    setSaving(true);
    let saved;
    try {
      saved = await updateSettings({ [field]: next });
    } catch {
      // Already recorded by `useUserSettings`; main must not be told of a
      // preference that did not persist.
      return;
    } finally {
      setSaving(false);
    }
    notifyUpdatePreferences(updatePreferencesFrom(saved));
  };

  const showHandedOffNote = installMayStillHappen({
    handedOff,
    preferences: resolveUpdatePreferences(stored),
  });

  return (
    <SettingsCard
      icon={<RefreshCw className="w-5 h-5 text-[var(--fit-good)]" />}
      title={t("settings:updatePreferences.title")}
      description={t("settings:updatePreferences.description")}
    >
      <div className="space-y-4">
        {SWITCHES.map(({ key, field }) => {
          const label = t(`settings:updatePreferences.${key}.label`);
          return (
            <div key={key} className="flex items-start justify-between gap-6">
              <div>
                <p className="text-[13px] font-medium text-[var(--ink)]">{label}</p>
                <p className="text-[12px] text-[var(--ink-dim)] mt-1 max-w-md leading-relaxed">
                  {t(`settings:updatePreferences.${key}.help`)}
                </p>
              </div>
              <ToggleSwitch
                checked={stored[key]}
                onChange={(next) => handleToggle(field, next)}
                label={label}
                disabled={loading || saving || (key !== "check" && !stored.check)}
              />
            </div>
          );
        })}
        {!stored.check && (
          <p className="text-[12px] text-[var(--ink-faint)] leading-relaxed">
            {t("settings:updatePreferences.checkOffNote")}
          </p>
        )}
        {showHandedOffNote && (
          <p className="text-[12px] text-[var(--fit-heavy)] leading-relaxed">
            {t("settings:updatePreferences.handedOffNote")}
          </p>
        )}
      </div>
    </SettingsCard>
  );
}

UpdatePreferencesCard.propTypes = {
  /** The user settings (`useUserSettings`), null while loading or unreadable. */
  settings: PropTypes.shape({
    auto_update_enabled: PropTypes.bool,
    auto_download_updates: PropTypes.bool,
    auto_install_updates: PropTypes.bool,
  }),
  loading: PropTypes.bool.isRequired,
  /** `useUserSettings().updateSettings`: PUTs a partial payload, resolves to the saved settings. */
  updateSettings: PropTypes.func.isRequired,
};
