import React from "react";
import PropTypes from "prop-types";
import ConfigPopup from "./modals/ConfigPopup";
import ConfigGlobalPopup from "./modals/ConfigGlobalPopup";
import EffortPopup from "./modals/EffortPopup";
import ModelPopup from "./modals/ModelPopup";

/**
 * The popups the slash commands open (#585), one at a time: `command` is the
 * `{ name, arg }` being run (from the composer or the header panel), or null.
 * Every popup reads and writes the conversation through `settings` (the
 * useConversationSettings owner); /model hands the picked model to the page.
 */
export default function CommandPopups({ command, onClose, settings, model, models, onPickModel }) {
  const isOpen = (name, arg = null) => command?.name === name && (command.arg ?? null) === arg;
  const open = {
    config: isOpen("config"),
    configGlobal: isOpen("config", "global"),
    effort: isOpen("effort"),
    model: isOpen("model"),
  };
  return (
    <>
      <ConfigPopup isOpen={open.config} onClose={onClose} settings={settings} model={model} />
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
    </>
  );
}

CommandPopups.propTypes = {
  command: PropTypes.shape({ name: PropTypes.string.isRequired, arg: PropTypes.string }),
  onClose: PropTypes.func.isRequired,
  settings: PropTypes.object.isRequired,
  model: PropTypes.object,
  models: PropTypes.array.isRequired,
  onPickModel: PropTypes.func.isRequired,
};
