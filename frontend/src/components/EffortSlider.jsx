import React from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import { REASONING_EFFORT_LEVELS } from "../utils/reasoningEffort";

/**
 * The reasoning effort as a 5-notch graduated slider (#609): positions 0-4
 * are the levels none, low, medium, high, xhigh, each notch labelled. A null
 * value (not known yet) renders the slider disabled.
 */
export default function EffortSlider({ value, onChange, disabled = false, label }) {
  const { t } = useTranslation();
  const index = Math.max(0, REASONING_EFFORT_LEVELS.indexOf(value));
  const levelLabel = (level) => t(`chat:config.effort.levels.${level}`);
  const pct = (index / (REASONING_EFFORT_LEVELS.length - 1)) * 100;

  return (
    <div className="w-full">
      <input
        type="range"
        min="0"
        max={REASONING_EFFORT_LEVELS.length - 1}
        step="1"
        value={index}
        disabled={disabled || value === null}
        aria-label={label}
        aria-valuetext={value === null ? undefined : levelLabel(value)}
        onChange={(e) => onChange(REASONING_EFFORT_LEVELS[Number(e.target.value)])}
        className="erudi-range w-full rounded-full cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
        style={{
          background: `linear-gradient(to right, #25C08A 0%, #1EAB78 ${pct}%, rgba(255,255,255,0.06) ${pct}%, rgba(255,255,255,0.06) 100%)`,
        }}
      />
      <div className="mt-1.5 flex justify-between text-[11px] text-gray-400" aria-hidden>
        {REASONING_EFFORT_LEVELS.map((level, i) => (
          <span
            key={level}
            className={i === index && value !== null ? "font-semibold text-emerald-300" : ""}
          >
            {levelLabel(level)}
          </span>
        ))}
      </div>
    </div>
  );
}

EffortSlider.propTypes = {
  value: PropTypes.oneOf(REASONING_EFFORT_LEVELS),
  onChange: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
  label: PropTypes.string.isRequired,
};
