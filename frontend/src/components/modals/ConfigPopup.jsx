import React, { useEffect, useId, useState } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import ModalShell from "./ModalShell";
import HelpTip from "../HelpTip";
import ToggleSwitch from "../ToggleSwitch";
import EffortSlider from "../EffortSlider";
import { formatNumber } from "../../i18n/format";
import { COMPACTION_THRESHOLDS, formatCompactionThreshold } from "../../utils/compactionThreshold";
import { webSearchUnavailabilityReason } from "../../utils/modelCapabilities";
import { hasNoPublisherRecommendation } from "../../utils/samplingDefaults";

// Creativity / diversity read-outs: two decimals in the active locale.
const TWO_DECIMALS = { minimumFractionDigits: 2, maximumFractionDigits: 2 };

// What a read-out shows while its value is not known yet.
export const NO_VALUE = "\u2013";

const WEB_SEARCH_REASON_KEYS = {
  incapable: "chat:config.webSearch.unavailable",
  unverified: "chat:config.webSearch.unverified",
};

const fillTo = (value, max) => {
  const pct = Math.round(((value ?? 0) / max) * 100);
  return {
    background: `linear-gradient(to right, #25C08A 0%, #1EAB78 ${pct}%, rgba(255,255,255,0.06) ${pct}%, rgba(255,255,255,0.06) 100%)`,
  };
};

/** A field row: its label, the `?` beside it, and the control. */
export function Field({ label, help, labelFor, aside, children }) {
  const { t } = useTranslation();
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1.5">
        <label
          htmlFor={labelFor}
          className="text-[0.72rem] uppercase tracking-wide font-semibold text-gray-300/80"
        >
          {label}
        </label>
        <HelpTip label={t("chat:config.help", { field: label })} text={help} />
        {aside && <div className="ml-auto">{aside}</div>}
      </div>
      {children}
    </div>
  );
}

Field.propTypes = {
  label: PropTypes.string.isRequired,
  help: PropTypes.string.isRequired,
  labelFor: PropTypes.string,
  aside: PropTypes.node,
  children: PropTypes.node,
};

/** The failed-save line every settings popup shows. */
export function SaveFailedNotice({ shown }) {
  const { t } = useTranslation();
  if (!shown) return null;
  return (
    <p role="alert" className="mt-4 text-xs text-red-300/90">
      {t("chat:config.saveFailed")}
    </p>
  );
}

SaveFailedNotice.propTypes = { shown: PropTypes.bool };

/**
 * /config (#609): every adjustable value of the conversation, or of the draft
 * on the new-conversation page, each saved through the settings hook as it
 * changes. No Apply button, no section for new-conversation defaults (that is
 * /config global).
 */
export default function ConfigPopup({ isOpen, onClose, settings, model }) {
  const { t } = useTranslation();
  const { values, setField, saveFailed, isDraft } = settings;
  const ids = {
    temperature: useId(),
    topP: useId(),
    threshold: useId(),
    instructions: useId(),
  };

  // The instructions are edited locally and saved on Save.
  const [instructions, setInstructions] = useState(values.custom_prompt ?? "");
  useEffect(() => {
    setInstructions(values.custom_prompt ?? "");
  }, [values.custom_prompt, isOpen]);

  const webSearchReason = webSearchUnavailabilityReason(model);
  const webSearchHelp = webSearchReason
    ? t(WEB_SEARCH_REASON_KEYS[webSearchReason])
    : t("chat:config.webSearch.help");

  const setEffort = (level) => setField("reasoning_effort", level);
  const setTemperature = (e) => setField("temperature", parseFloat(e.target.value));
  const setTopP = (e) => setField("top_p", parseFloat(e.target.value));
  const setWebSearch = (next) => setField("web_search_enabled", next);
  const setThreshold = (e) => setField("compaction_threshold", Number(e.target.value));
  // The instructions are saved on Save, when the editor loses the focus, and
  // when the popup closes with an edit in it: never lost on close.
  const edited = values.custom_prompt !== null && instructions !== values.custom_prompt;
  const saveInstructions = () => {
    if (edited) setField("custom_prompt", instructions);
  };
  const close = () => {
    saveInstructions();
    onClose();
  };

  const effortLabel = t("chat:config.effort.label");
  const webSearchLabel = t("chat:config.webSearch.label");
  const creativityLabel = t("chat:config.creativity.label");
  const diversityLabel = t("chat:config.diversity.label");
  const thresholdLabel = t("chat:config.threshold.label");
  const instructionsLabel = t("chat:config.instructions.label");

  return (
    <ModalShell
      isOpen={isOpen}
      onClose={close}
      title={t("chat:config.title")}
      subtitle={isDraft ? t("chat:config.subtitleDraft") : t("chat:config.subtitle")}
      width="max-w-2xl"
    >
      <div className="grid gap-6 md:grid-cols-2">
        <div className="flex flex-col gap-6">
          <Field label={effortLabel} help={t("chat:config.effort.help")}>
            <EffortSlider
              value={values.reasoning_effort}
              onChange={setEffort}
              label={effortLabel}
            />
          </Field>

          <Field
            label={creativityLabel}
            help={t("chat:config.creativity.help")}
            labelFor={ids.temperature}
            aside={
              <span
                className={`text-[11px] font-semibold px-2 py-0.5 rounded-md border ${
                  values.temperature > 1
                    ? "text-amber-200/90 bg-amber-500/10 border-amber-400/30"
                    : "text-emerald-200/90 bg-emerald-500/10 border-emerald-400/25"
                }`}
              >
                {values.temperature === null
                  ? NO_VALUE
                  : formatNumber(values.temperature, TWO_DECIMALS)}
              </span>
            }
          >
            <input
              id={ids.temperature}
              type="range"
              min="0"
              max="2"
              step="0.01"
              aria-label={creativityLabel}
              value={values.temperature ?? 0}
              disabled={values.temperature === null}
              onChange={setTemperature}
              className="erudi-range w-full rounded-full cursor-pointer disabled:opacity-50"
              style={fillTo(values.temperature, 2)}
            />
          </Field>

          <Field
            label={diversityLabel}
            help={t("chat:config.diversity.help")}
            labelFor={ids.topP}
            aside={
              <span className="text-[11px] font-semibold text-emerald-200/90 bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-400/25">
                {values.top_p === null ? NO_VALUE : formatNumber(values.top_p, TWO_DECIMALS)}
              </span>
            }
          >
            <input
              id={ids.topP}
              type="range"
              min="0"
              max="1"
              step="0.01"
              aria-label={diversityLabel}
              value={values.top_p ?? 0}
              disabled={values.top_p === null}
              onChange={setTopP}
              className="erudi-range w-full rounded-full cursor-pointer disabled:opacity-50"
              style={fillTo(values.top_p, 1)}
            />
          </Field>
          {hasNoPublisherRecommendation(model) && (
            <p className="-mt-3 text-[11px] leading-snug text-gray-400/80">
              {t("chat:config.noPublisherRecommendation")}
            </p>
          )}
        </div>

        <div className="flex flex-col gap-6">
          <Field
            label={webSearchLabel}
            help={webSearchHelp}
            aside={
              <ToggleSwitch
                checked={values.web_search_enabled === true}
                onChange={setWebSearch}
                label={webSearchLabel}
                disabled={webSearchReason !== null || values.web_search_enabled === null}
              />
            }
          />

          <Field
            label={thresholdLabel}
            help={t("chat:config.threshold.help")}
            labelFor={ids.threshold}
            aside={
              <select
                id={ids.threshold}
                aria-label={thresholdLabel}
                value={values.compaction_threshold ?? ""}
                disabled={values.compaction_threshold === null}
                onChange={setThreshold}
                className="text-[11px] font-semibold rounded-md border border-emerald-400/25 bg-emerald-500/10 text-emerald-200/90 px-2 py-1 focus:outline-none focus:border-emerald-400/50 transition-colors cursor-pointer"
              >
                {COMPACTION_THRESHOLDS.map((threshold) => (
                  <option key={threshold} value={threshold}>
                    {formatCompactionThreshold(threshold)}
                  </option>
                ))}
              </select>
            }
          />

          <Field
            label={instructionsLabel}
            help={t("chat:config.instructions.help")}
            labelFor={ids.instructions}
          >
            <textarea
              id={ids.instructions}
              aria-label={instructionsLabel}
              value={instructions}
              disabled={values.custom_prompt === null}
              onChange={(e) => setInstructions(e.target.value)}
              onBlur={saveInstructions}
              placeholder={t("chat:config.instructions.placeholder")}
              className={[
                "w-full h-28 rounded-2xl resize-none",
                "bg-white/10 border border-white/20",
                "px-4 py-3 text-sm text-gray-100 placeholder-gray-400",
                "focus:outline-none focus:ring-2 focus:ring-emerald-400/60 focus:border-emerald-400/60",
                "custom-scroll",
              ].join(" ")}
            />
            <div className="mt-2 flex justify-end">
              <button
                type="button"
                onClick={saveInstructions}
                disabled={values.custom_prompt === null || instructions === values.custom_prompt}
                className="rounded-full px-4 py-1.5 text-sm font-semibold bg-emerald-500 hover:bg-emerald-600 text-[#0f2f25] border border-white/20 shadow transition active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {t("chat:config.instructions.save")}
              </button>
            </div>
          </Field>
        </div>
      </div>
      <SaveFailedNotice shown={saveFailed} />
    </ModalShell>
  );
}

ConfigPopup.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  settings: PropTypes.shape({
    values: PropTypes.object.isRequired,
    setField: PropTypes.func.isRequired,
    saveFailed: PropTypes.bool,
    isDraft: PropTypes.bool,
  }).isRequired,
  model: PropTypes.object,
};
