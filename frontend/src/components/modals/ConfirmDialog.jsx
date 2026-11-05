import React from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import ModalShell from "./ModalShell";

/**
 * A confirmation in the command popups' shell (#585): a title, what
 * confirming does, and Cancel / confirm. Cancel comes first and takes the
 * focus, so Enter right away backs out; Escape, the close button and a click
 * outside back out too.
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
          onClick={onConfirm}
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
