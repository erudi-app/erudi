import React, { useEffect, useRef } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import ModalShell from "./ModalShell";

/**
 * A confirmation in the command popups' shell (#585): a title, what
 * confirming does, and Cancel / confirm. Cancel comes first and takes the
 * focus, so Enter right away backs out; Escape, the close button and a click
 * outside back out too. It confirms once per opening: a second click, while
 * the dialog plays its closing animation (still on screen, still clickable),
 * does nothing.
 */
export default function ConfirmDialog({
  isOpen,
  title,
  message,
  confirmLabel,
  onConfirm,
  onClose,
}) {
  const { t } = useTranslation();
  // A ref, not state: the closing dialog keeps the handlers of its last
  // render, and only a ref is read fresh by them.
  const confirmedRef = useRef(false);
  const openRef = useRef(isOpen);
  useEffect(() => {
    openRef.current = isOpen;
    if (isOpen) confirmedRef.current = false;
  }, [isOpen]);

  const confirm = () => {
    if (!openRef.current || confirmedRef.current) return;
    confirmedRef.current = true;
    onConfirm();
  };

  return (
    <ModalShell isOpen={isOpen} onClose={onClose} title={title} subtitle={message}>
      <div className="flex justify-end gap-2">
        <button
          type="button"
          onClick={onClose}
          className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 text-gray-100 transition"
        >
          {t("common:actions.cancel")}
        </button>
        <button
          type="button"
          onClick={confirm}
          className="rounded-xl px-4 py-2 text-sm font-semibold bg-emerald-500/80 hover:bg-emerald-500 border border-emerald-400/40 text-white transition"
        >
          {confirmLabel}
        </button>
      </div>
    </ModalShell>
  );
}

ConfirmDialog.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  title: PropTypes.string.isRequired,
  message: PropTypes.string.isRequired,
  confirmLabel: PropTypes.string.isRequired,
  onConfirm: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired,
};
