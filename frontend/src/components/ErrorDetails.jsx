import React, { useId, useState } from "react";
import PropTypes from "prop-types";
import { Check, ChevronDown, ChevronRight, Copy } from "lucide-react";
import { useTranslation } from "react-i18next";

// The discreet "Details" control under a generic failure (#615).
//
// Collapsed by default. Opened, it quotes the cleaned cause exactly as the
// backend wrote it (never translated: it is what a bug report needs) and the
// id of the failing request, copyable, which is the id of the matching entry
// in Diagnostics. One button copies both, for a bug report. The backend uses "-" for a request id it never had; that is
// not an id worth showing.

const NO_REQUEST_ID = "-";

export default function ErrorDetails({ detail }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const panelId = useId();
  const requestId =
    detail.request_id && detail.request_id !== NO_REQUEST_ID ? detail.request_id : null;

  const copyDetails = () => {
    const text = requestId ? `${detail.text}\nrequest_id: ${requestId}` : detail.text;
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1000);
      })
      .catch(() => {
        // The clipboard refused (permission, focus): nothing is lost, the text
        // stays selectable on screen, so the control just does not tick.
      });
  };

  return (
    <div className="flex flex-col gap-1 text-xs text-[var(--ink-faint)]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        className="flex w-fit items-center gap-1 transition-colors hover:text-[var(--ink)]"
      >
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        )}
        <span>{open ? t("chat:errorDetails.hide") : t("chat:errorDetails.show")}</span>
      </button>
      {open && (
        <div
          id={panelId}
          className="flex flex-col gap-1 rounded-xl border border-[var(--line-strong)] bg-white/[0.03] px-3 py-2 text-[var(--ink-dim)]"
        >
          <span className="break-words font-mono">{detail.text}</span>
          {requestId && (
            <span className="flex items-center gap-2">
              <span>{t("chat:errorDetails.requestId")}</span>
              <span className="font-mono">{requestId}</span>
            </span>
          )}
          <button
            type="button"
            onClick={copyDetails}
            aria-label={t("chat:errorDetails.copy")}
            title={t("chat:errorDetails.copy")}
            className="w-fit transition-colors hover:text-[var(--ink)]"
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-green-400" aria-hidden="true" />
            ) : (
              <Copy className="h-3.5 w-3.5" aria-hidden="true" />
            )}
          </button>
        </div>
      )}
    </div>
  );
}

ErrorDetails.propTypes = {
  // The failed turn's {"t":"error_detail","text","request_id"} trace event.
  detail: PropTypes.shape({
    text: PropTypes.string.isRequired,
    request_id: PropTypes.string,
  }).isRequired,
};
