import React, { useCallback, useEffect, useState } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import ModalShell from "./ModalShell";
import { NO_VALUE } from "./ConfigPopup";
import { formatNumber } from "../../i18n/format";
import { formatCompactionThreshold } from "../../utils/compactionThreshold";
import { createLogger } from "../../utils/logger";

const log = createLogger("ConfigGlobalPopup");

const TWO_DECIMALS = { minimumFractionDigits: 2, maximumFractionDigits: 2 };

/**
 * /config global (#609): saves the six current values as the defaults of the
 * current model the moment it opens, then says so and lists what was saved.
 * Empty instructions clear the instructions saved for the model.
 */
export default function ConfigGlobalPopup({ isOpen, onClose, settings, model }) {
  const { t } = useTranslation();
  const [status, setStatus] = useState("saving");
  const { saveAsDefaults, values } = settings;
  const modelName = model?.name ?? "";

  const save = useCallback(async () => {
    try {
      await saveAsDefaults();
      setStatus("saved");
    } catch (error) {
      log.error(`Failed to save the conversation defaults of model ${model?.id}`, error);
      setStatus("failed");
    }
  }, [saveAsDefaults, model?.id]);

  // Each opening starts on "saving": set while rendering, before the save runs.
  const [wasOpen, setWasOpen] = useState(false);
  if (wasOpen !== isOpen) {
    setWasOpen(isOpen);
    if (isOpen) setStatus("saving");
  }

  useEffect(() => {
    // save sets state only once the request has answered (the rule does not
    // see the await); "saving" was set while rendering, above.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (isOpen) save();
    // Saved once per opening: a re-render of the page must not save again.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  // A value not known yet reads as a dash, never as a made-up one.
  const shown = (value, format) =>
    value === null || value === undefined ? NO_VALUE : format(value);
  const rows = [
    [
      t("chat:config.effort.label"),
      shown(values.reasoning_effort, (level) => t(`chat:config.effort.levels.${level}`)),
    ],
    [
      t("chat:config.webSearch.label"),
      shown(values.web_search_enabled, (on) =>
        on ? t("chat:config.global.on") : t("chat:config.global.off")
      ),
    ],
    [
      t("chat:config.creativity.label"),
      shown(values.temperature, (value) => formatNumber(value, TWO_DECIMALS)),
    ],
    [
      t("chat:config.diversity.label"),
      shown(values.top_p, (value) => formatNumber(value, TWO_DECIMALS)),
    ],
    [
      t("chat:config.threshold.label"),
      shown(values.compaction_threshold, formatCompactionThreshold),
    ],
    [
      t("chat:config.instructions.label"),
      shown(values.custom_prompt, (text) => text || t("chat:config.global.noInstructions")),
    ],
  ];

  return (
    <ModalShell
      isOpen={isOpen}
      onClose={onClose}
      title={t("chat:config.global.title")}
      subtitle={t("chat:config.global.subtitle", { model: modelName })}
    >
      {status === "saving" && (
        <p className="text-sm text-gray-300" aria-live="polite">
          {t("chat:config.global.saving", { model: modelName })}
        </p>
      )}
      {status === "saved" && (
        <p className="text-sm font-semibold text-emerald-300" aria-live="polite">
          {t("chat:config.global.saved", { model: modelName })}
        </p>
      )}
      {status === "failed" && (
        <div className="flex items-center justify-between gap-3">
          <p role="alert" className="text-sm text-red-300/90">
            {t("chat:config.global.failed", { model: modelName })}
          </p>
          <button
            type="button"
            onClick={save}
            className="rounded-full px-4 py-1.5 text-sm font-semibold bg-white/10 hover:bg-white/15 text-gray-100 border border-white/20 transition active:scale-95"
          >
            {t("common:actions.retry")}
          </button>
        </div>
      )}
      <dl className="mt-5 grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
        {rows.map(([label, value]) => (
          <React.Fragment key={label}>
            <dt className="text-gray-400">{label}</dt>
            <dd className="text-gray-100 truncate" title={value}>
              {value}
            </dd>
          </React.Fragment>
        ))}
      </dl>
    </ModalShell>
  );
}

ConfigGlobalPopup.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  settings: PropTypes.shape({
    values: PropTypes.object.isRequired,
    saveAsDefaults: PropTypes.func.isRequired,
  }).isRequired,
  model: PropTypes.shape({ id: PropTypes.number, name: PropTypes.string }),
};
