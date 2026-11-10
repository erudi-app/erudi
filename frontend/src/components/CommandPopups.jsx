import React from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import ConfigPopup from "./modals/ConfigPopup";
import ConfigGlobalPopup from "./modals/ConfigGlobalPopup";
import ConfirmDialog from "./modals/ConfirmDialog";
import EffortPopup from "./modals/EffortPopup";
import ModelPopup from "./modals/ModelPopup";

// /config global, which the settings window's Save as defaults button runs.
const CONFIG_GLOBAL = { name: "config", arg: "global" };

/**
 * The popups the slash commands open (#585), one at a time: `command` is the
 * `{ name, arg }` being run (from the composer or the header panel), or null.
 * Every popup reads and writes the conversation through `settings` (the
 * useConversationSettings owner); /model hands the picked model to the page,
 * and /compact's confirmation hands the go-ahead to it (`onCompact`, on a
 * conversation only). /config's "Save as defaults" button runs /config global
 * through `onCommand` (the page's own command runner), unless
 * `saveDefaultsDisabled` (an answer is being written, a compaction runs).
 */
export default function CommandPopups({
  command,
  onClose,
  onCommand,
  saveDefaultsDisabled = false,
  settings,
  model,
  models,
  onPickModel,
  onCompact,
}) {
  const { t } = useTranslation();
  const isOpen = (name, arg = null) => command?.name === name && (command.arg ?? null) === arg;
  const open = {
    config: isOpen("config"),
    configGlobal: isOpen("config", "global"),
    effort: isOpen("effort"),
    model: isOpen("model"),
    compact: isOpen("compact"),
  };
  return (
    <>
      <ConfigPopup
        isOpen={open.config}
        onClose={onClose}
        settings={settings}
        model={model}
        onSaveAsDefaults={() => onCommand(CONFIG_GLOBAL)}
        saveAsDefaultsDisabled={saveDefaultsDisabled}
      />
      <ConfigGlobalPopup
        isOpen={open.configGlobal}
        onClose={onClose}
        settings={settings}
        model={model}
      />
      <EffortPopup isOpen={open.effort} onClose={onClose} settings={settings} />
      <ModelPopup
        isOpen={open.model}
        onClose={onClose}
        models={models}
        currentId={settings.values.llm_id}
        onPick={onPickModel}
      />
      {onCompact && (
        <ConfirmDialog
          isOpen={open.compact}
          title={t("chat:compactDialog.title")}
          message={t("chat:compactDialog.body")}
          confirmLabel={t("chat:compactDialog.confirm")}
          onConfirm={() => {
            onClose();
            onCompact();
          }}
          onClose={onClose}
        />
      )}
    </>
  );
}

CommandPopups.propTypes = {
  command: PropTypes.shape({ name: PropTypes.string.isRequired, arg: PropTypes.string }),
  onClose: PropTypes.func.isRequired,
  onCommand: PropTypes.func.isRequired,
  saveDefaultsDisabled: PropTypes.bool,
  settings: PropTypes.object.isRequired,
  model: PropTypes.object,
  models: PropTypes.array.isRequired,
  onPickModel: PropTypes.func.isRequired,
  onCompact: PropTypes.func,
};
