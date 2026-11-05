import React from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import ModalShell from "./ModalShell";
import EffortSlider from "../EffortSlider";
import { Field, SaveFailedNotice } from "./ConfigPopup";

/** /effort (#609): the reasoning effort slider alone, saved as it changes. */
export default function EffortPopup({ isOpen, onClose, settings }) {
  const { t } = useTranslation();
  const { values, setField, saveFailed, isDraft } = settings;
  const label = t("chat:config.effort.label");
  const setEffort = (level) => setField("reasoning_effort", level);
  return (
    <ModalShell
      isOpen={isOpen}
      onClose={onClose}
      title={label}
      subtitle={isDraft ? t("chat:effortPopup.subtitleDraft") : t("chat:effortPopup.subtitle")}
    >
      <Field label={label} help={t("chat:config.effort.help")}>
        <EffortSlider value={values.reasoning_effort} onChange={setEffort} label={label} />
      </Field>
      <SaveFailedNotice shown={saveFailed} />
    </ModalShell>
  );
}

EffortPopup.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  settings: PropTypes.shape({
    values: PropTypes.object.isRequired,
    setField: PropTypes.func.isRequired,
    saveFailed: PropTypes.bool,
    isDraft: PropTypes.bool,
  }).isRequired,
};
