import React, { useEffect, useRef } from "react";
import PropTypes from "prop-types";
import { X } from "lucide-react";
import { useTranslation } from "react-i18next";

// How long an informative notice stays before it goes on its own.
const INFO_NOTICE_MS = 6000;

/**
 * The one-line outcome of a command above the composer (#585): an
 * informative notice ("Nothing to compact yet.") that goes after a few
 * seconds, or an error that stays until it is closed or the next command.
 */
export default function CommandNotice({ notice, onDismiss }) {
  const { t } = useTranslation();
  const onDismissRef = useRef(onDismiss);
  useEffect(() => {
    onDismissRef.current = onDismiss;
  });

  useEffect(() => {
    if (notice?.tone !== "info") return undefined;
    const timer = setTimeout(() => onDismissRef.current(), INFO_NOTICE_MS);
    return () => clearTimeout(timer);
  }, [notice]);

  if (!notice) return null;
  const isError = notice.tone === "error";
  return (
    <div
      role={isError ? "alert" : "status"}
      className={[
        "mb-3 flex items-center justify-between gap-3 px-3 py-2 rounded-lg border text-xs",
        isError
          ? "bg-red-500/10 border-red-500/30 text-red-200"
          : "bg-white/5 border-white/15 text-gray-200",
      ].join(" ")}
    >
      <span>{notice.text}</span>
      <button
        type="button"
        onClick={onDismiss}
        aria-label={t("common:actions.close")}
        className="shrink-0 rounded-md p-1 hover:bg-white/10 transition"
      >
        <X className="w-3.5 h-3.5" aria-hidden />
      </button>
    </div>
  );
}

CommandNotice.propTypes = {
  // { tone: "info" | "error", text }, or null for none.
  notice: PropTypes.shape({
    tone: PropTypes.oneOf(["info", "error"]).isRequired,
    text: PropTypes.string.isRequired,
  }),
  onDismiss: PropTypes.func.isRequired,
};
