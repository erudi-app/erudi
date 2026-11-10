import React, { useEffect, useId, useRef, useState } from "react";
import PropTypes from "prop-types";
import { Check, ChevronDown } from "lucide-react";
import { useTranslation } from "react-i18next";
import { offeredModels } from "../utils/modelWeights";

/**
 * The model name of the conversation header as a picker (#704), on a
 * conversation and on the new-conversation page. A button showing the name
 * (red "Please select a model" when `attention`: the orphaned conversation)
 * that opens the installed models as a listbox. The list is the one the Model
 * popup offers (`offeredModels`); picking a model other than `currentId` hands
 * it to `onPick`, which the page runs exactly as it runs the Model command's
 * pick. When no model can be offered the open menu says how to get one.
 *
 * Keyboard: Enter, Space, ArrowDown and ArrowUp on the button open it; the
 * arrows, Home and End move, Enter and Space pick, Escape closes and returns
 * the focus to the button, Tab closes. `disabled` closes it too.
 */
export default function ModelPicker({
  modelName,
  attention = false,
  models,
  currentId,
  onPick,
  disabled = false,
}) {
  const { t } = useTranslation();
  const baseId = useId();
  const listId = `${baseId}-list`;
  const optionId = (index) => `${baseId}-option-${index}`;
  const rootRef = useRef(null);
  const triggerRef = useRef(null);
  const popupRef = useRef(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);

  const offered = offeredModels(models);
  // A picker that becomes disabled (an answer starts, a compaction runs)
  // closes: adjusted while rendering, so no render shows it open.
  if (open && disabled) setOpen(false);
  const isOpen = open && !disabled;

  const openMenu = (at = null) => {
    const current = offered.findIndex((model) => model.id === currentId);
    setActive(at ?? Math.max(current, 0));
    setOpen(true);
  };
  const closeMenu = (returnFocus) => {
    setOpen(false);
    if (returnFocus) triggerRef.current?.focus();
  };
  const pick = (model) => {
    if (model.id !== currentId) onPick(model);
    closeMenu(true);
  };

  // The focus moves into the menu on open (the listbox, or the message that
  // says how to get a model), where the keys are handled.
  useEffect(() => {
    if (isOpen) popupRef.current?.focus();
  }, [isOpen]);

  // A press outside closes it, the focus staying where the user put it.
  useEffect(() => {
    if (!isOpen) return undefined;
    const onPress = (event) => {
      if (rootRef.current && !rootRef.current.contains(event.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onPress);
    return () => document.removeEventListener("mousedown", onPress);
  }, [isOpen]);

  const onTriggerKeyDown = (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!isOpen) openMenu();
    }
  };

  const onPopupKeyDown = (event) => {
    const last = offered.length - 1;
    switch (event.key) {
      case "Escape":
        event.preventDefault();
        event.stopPropagation();
        closeMenu(true);
        break;
      case "Tab":
        closeMenu(false);
        break;
      case "ArrowDown":
        event.preventDefault();
        setActive((index) => Math.min(index + 1, last));
        break;
      case "ArrowUp":
        event.preventDefault();
        setActive((index) => Math.max(index - 1, 0));
        break;
      case "Home":
        event.preventDefault();
        setActive(0);
        break;
      case "End":
        event.preventDefault();
        setActive(last);
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        if (offered[active]) pick(offered[active]);
        break;
      default:
    }
  };

  return (
    <span ref={rootRef} className="relative inline-block max-w-full align-bottom">
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={isOpen ? listId : undefined}
        onClick={() => (isOpen ? closeMenu(false) : openMenu())}
        onKeyDown={onTriggerKeyDown}
        className={[
          "inline-flex max-w-full items-center gap-1 rounded-lg px-1.5 -mx-1.5 align-bottom",
          "transition hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60",
          "disabled:cursor-not-allowed disabled:hover:bg-transparent",
        ].join(" ")}
      >
        <span
          className={`truncate ${attention ? "text-red-400" : "text-emerald-300"}`}
          title={attention ? undefined : modelName}
        >
          {attention ? t("chat:conversationHeader.pickerAttention") : modelName}
        </span>
        <ChevronDown
          aria-hidden
          className={`w-4 h-4 shrink-0 transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
      </button>

      {isOpen && (
        <div
          onKeyDown={onPopupKeyDown}
          className={[
            "absolute left-0 top-full z-30 mt-2 w-72 max-w-[80vw] rounded-xl p-1.5",
            "border border-white/10 bg-[rgba(14,28,25,0.95)] backdrop-blur-[18px] shadow-lg",
            "text-sm font-normal tracking-normal",
          ].join(" ")}
        >
          {offered.length === 0 ? (
            <p ref={popupRef} tabIndex={-1} className="px-3 py-2 text-gray-300 focus:outline-none">
              {t("chat:modelPicker.empty")}
            </p>
          ) : (
            <ul
              ref={popupRef}
              id={listId}
              role="listbox"
              tabIndex={-1}
              aria-label={t("chat:modelPicker.listLabel")}
              aria-activedescendant={optionId(active)}
              className="max-h-64 overflow-y-auto custom-scroll focus:outline-none"
            >
              {offered.map((model, index) => {
                const current = model.id === currentId;
                return (
                  <li
                    key={model.id}
                    id={optionId(index)}
                    role="option"
                    aria-selected={current}
                    onMouseMove={() => setActive(index)}
                    onClick={() => pick(model)}
                    className={[
                      "flex cursor-pointer items-center justify-between gap-3 rounded-lg px-3 py-2",
                      index === active ? "bg-white/10" : "",
                      current ? "text-emerald-200" : "text-gray-100",
                    ].join(" ")}
                  >
                    <span className="truncate" title={model.name}>
                      {model.name}
                    </span>
                    {current && (
                      <span className="inline-flex shrink-0 items-center gap-1 text-xs text-emerald-300">
                        <Check className="w-3.5 h-3.5" aria-hidden />
                        {t("chat:modelPicker.current")}
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </span>
  );
}

ModelPicker.propTypes = {
  modelName: PropTypes.string,
  attention: PropTypes.bool,
  models: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      name: PropTypes.string.isRequired,
    })
  ).isRequired,
  currentId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  onPick: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
};
