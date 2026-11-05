import React from "react";
import PropTypes from "prop-types";
import { Check } from "lucide-react";
import { useTranslation } from "react-i18next";
import ModalShell from "./ModalShell";
import { Field } from "./ConfigPopup";
import { hasMissingWeights } from "../../utils/modelWeights";

/**
 * /model (#585): the installed models a conversation can run on. A model
 * whose weights are missing (an orphaned knowledge-base assistant, #376) is
 * not offered. Picking the current model only closes the popup.
 */
export default function ModelPopup({ isOpen, onClose, models, currentId, onPick }) {
  const { t } = useTranslation();
  const offered = models.filter((model) => !hasMissingWeights(model));
  const label = t("chat:modelPopup.listLabel");

  const pick = (model) => {
    if (model.id !== currentId) onPick(model);
    onClose();
  };

  return (
    <ModalShell
      isOpen={isOpen}
      onClose={onClose}
      title={t("chat:modelPopup.title")}
      subtitle={t("chat:modelPopup.subtitle")}
    >
      <Field label={label} help={t("chat:modelPopup.help")}>
        {offered.length === 0 ? (
          <p className="text-sm text-gray-400">{t("chat:modelPopup.empty")}</p>
        ) : (
          <ul
            aria-label={label}
            className="flex flex-col gap-1.5 max-h-80 overflow-y-auto custom-scroll"
          >
            {offered.map((model) => {
              const current = model.id === currentId;
              return (
                <li key={model.id}>
                  <button
                    type="button"
                    aria-current={current ? "true" : undefined}
                    onClick={() => pick(model)}
                    className={[
                      "w-full flex items-center justify-between gap-3 rounded-xl px-4 py-2.5 text-left text-sm",
                      "border transition",
                      current
                        ? "border-emerald-400/40 bg-emerald-500/10 text-emerald-100"
                        : "border-white/10 bg-white/5 hover:bg-white/10 text-gray-100",
                    ].join(" ")}
                  >
                    <span className="truncate" title={model.name}>
                      {model.name}
                    </span>
                    {current && (
                      <span className="inline-flex shrink-0 items-center gap-1 text-xs text-emerald-300">
                        <Check className="w-3.5 h-3.5" aria-hidden />
                        {t("chat:modelPopup.current")}
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </Field>
    </ModalShell>
  );
}

ModelPopup.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  models: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      name: PropTypes.string.isRequired,
    })
  ).isRequired,
  currentId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  onPick: PropTypes.func.isRequired,
};
