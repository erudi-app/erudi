import React, { useId, useState } from "react";
import PropTypes from "prop-types";
import { motion, AnimatePresence } from "framer-motion";
import { SlidersHorizontal } from "lucide-react";
import { Trans, useTranslation } from "react-i18next";
import { headerCommands } from "../utils/slashCommands";

/**
 * The header of a conversation and of the new-conversation page (#585): a
 * static "You are chatting with <model>" line and a settings icon that opens
 * a panel with one button per command the page offers (`commands`, from the
 * slash-command registry, shown in its `headerOrder`), each running it
 * exactly as typing it does. What
 * the page adds as children renders in the panel after the command buttons.
 *
 * `attention` is the orphaned-conversation state (#225): the model is gone,
 * the line turns into a red "Please select a model", and the panel's Model
 * button stays usable although the composer is blocked.
 */
export default function ConversationHeader({
  modelName,
  attention = false,
  commands,
  disabledCommands = [],
  onCommand,
  saveFailed = false,
  children,
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const panelId = useId();

  return (
    <div
      className={[
        "relative w-full rounded-[26px]",
        "border border-white/10",
        "bg-[rgba(22,40,36,0.45)] backdrop-blur-[18px] saturate-[1.4]",
        "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)]",
      ].join(" ")}
    >
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none rounded-[26px] mix-blend-overlay"
        style={{
          background: "linear-gradient(to bottom, rgba(255,255,255,0.18), rgba(255,255,255,0) 40%)",
        }}
      />
      <div className="relative z-10 p-5">
        <div className="flex items-center justify-between gap-3">
          <h3 className="min-w-0 break-words text-[1.15rem] font-semibold tracking-tight text-[#F2F7F4]">
            {attention ? (
              <span role="alert" className="text-red-400">
                {t("chat:conversationHeader.pickerAttention")}
              </span>
            ) : (
              <Trans
                i18nKey="chat:conversationHeader.chattingWith"
                values={{ model: modelName }}
                components={{
                  model: <span className="text-emerald-300" title={modelName} />,
                }}
              />
            )}
          </h3>
          <button
            type="button"
            aria-label={t("chat:conversationHeader.settings")}
            aria-expanded={open}
            aria-controls={panelId}
            onClick={() => setOpen((v) => !v)}
            className={[
              "inline-flex items-center justify-center shrink-0 w-9 h-9 rounded-xl",
              "bg-white/5 hover:bg-white/10 border transition",
              attention
                ? "border-red-500/60 text-red-300"
                : "border-white/10 hover:border-white/20 text-gray-300 hover:text-emerald-400",
            ].join(" ")}
          >
            <SlidersHorizontal size={18} aria-hidden />
          </button>
        </div>
        {saveFailed && (
          <p role="alert" className="mt-2 text-xs text-red-300/90">
            {t("chat:config.saveFailed")}
          </p>
        )}

        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              id={panelId}
              key="commands"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ type: "tween", duration: 0.25 }}
              className="overflow-hidden"
            >
              <div className="mt-4 flex flex-wrap items-center gap-2">
                {headerCommands(commands).map((command) => (
                  <button
                    key={command.name}
                    type="button"
                    disabled={disabledCommands.includes(command.name)}
                    onClick={() => onCommand({ name: command.name, arg: null })}
                    title={t(`chat:commands.${command.name}.description`)}
                    className={[
                      "inline-flex items-baseline gap-2 rounded-xl px-3.5 py-2 text-sm font-semibold",
                      "bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20",
                      "text-gray-100 transition active:scale-95",
                      "disabled:opacity-40 disabled:cursor-not-allowed disabled:active:scale-100",
                    ].join(" ")}
                  >
                    {t(`chat:commands.${command.name}.label`)}
                    <span className="font-mono text-[11px] font-normal text-emerald-300/80">
                      {`/${command.name}`}
                    </span>
                  </button>
                ))}
                {children}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

ConversationHeader.propTypes = {
  modelName: PropTypes.string,
  attention: PropTypes.bool,
  commands: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      headerOrder: PropTypes.number.isRequired,
    })
  ).isRequired,
  disabledCommands: PropTypes.arrayOf(PropTypes.string),
  onCommand: PropTypes.func.isRequired,
  saveFailed: PropTypes.bool,
  children: PropTypes.node,
};
