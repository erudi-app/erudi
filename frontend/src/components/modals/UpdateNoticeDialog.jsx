import React, { useEffect, useId, useRef } from "react";
import PropTypes from "prop-types";
import { Sparkles } from "lucide-react";
import { useTranslation } from "react-i18next";
import { releasePageUrl } from "../../utils/updateNotice";

/**
 * "Updated to X.Y.Z" (#572), shown once over the app at the first launch of a
 * new version (`loadUpdateNotice` decides; App records the version as seen
 * when this is dismissed).
 *
 * The release notes, when they are this version's, arrive as text lines and
 * are rendered as text: no markup is interpreted and nothing is loaded. The
 * only link is the release page, which the window-open handler of main.js
 * hands to the system browser, and only on click. On the first launch over a
 * version that predates the update opt-ins, it also says downloading and
 * installing are now separate choices, both off (#563).
 */
export default function UpdateNoticeDialog({ notice, onDismiss }) {
  const { t } = useTranslation();
  const titleId = useId();
  const closeRef = useRef(null);

  useEffect(() => {
    if (!notice) return undefined;
    closeRef.current?.focus();
    const onKeyDown = (event) => {
      if (event.key === "Escape") onDismiss();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [notice, onDismiss]);

  if (!notice) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[9999] p-4">
      <div
        role="dialog"
        aria-modal={true}
        aria-labelledby={titleId}
        className={[
          "relative w-full max-w-lg max-h-[90vh] overflow-auto custom-scroll rounded-[26px]",
          "border border-white/10 bg-[rgba(22,40,36,0.92)] backdrop-blur-[18px] saturate-[1.4]",
          "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4)]",
        ].join(" ")}
      >
        <div className="p-6">
          <div className="flex items-start gap-3 mb-4">
            <div className="w-8 h-8 shrink-0 rounded-lg bg-[#34d399]/15 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-[#34d399]" />
            </div>
            <h2 id={titleId} className="text-xl font-semibold tracking-tight text-[#F2F7F4]">
              {t("common:updateNotice.title", { version: notice.version })}
            </h2>
          </div>

          {notice.lines ? (
            <div className="rounded-xl border border-white/10 bg-black/20 p-3 max-h-64 overflow-auto custom-scroll space-y-1">
              {notice.lines.map((line, index) => (
                // Lines repeat (two "Fixes" headings): the index is the identity.
                <p key={index} className="text-[13px] leading-relaxed text-[#cfd8d4]">
                  {line}
                </p>
              ))}
            </div>
          ) : (
            <p className="text-[13px] leading-relaxed text-[#cfd8d4]">
              {t("common:updateNotice.noNotes")}
            </p>
          )}

          {notice.explainOptIns && (
            <p className="mt-4 text-[13px] leading-relaxed text-[#cfd8d4]">
              {t("common:updateNotice.optIns")}
            </p>
          )}

          <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
            <a
              href={releasePageUrl(notice.version)}
              target="_blank"
              rel="noreferrer"
              className="text-[13px] text-[#e6efeb] hover:text-white underline underline-offset-2"
            >
              {t("common:updateNotice.releaseNotes")}
            </a>
            <button
              ref={closeRef}
              type="button"
              onClick={onDismiss}
              className="px-4 py-2 rounded-lg text-[13px] font-medium bg-[#34d399] text-[#02130e] transition-opacity hover:opacity-90"
            >
              {t("common:actions.close")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

UpdateNoticeDialog.propTypes = {
  /** The decision `loadUpdateNotice` returned, or null when there is nothing to show. */
  notice: PropTypes.shape({
    version: PropTypes.string.isRequired,
    lines: PropTypes.arrayOf(PropTypes.string),
    explainOptIns: PropTypes.bool,
  }),
  onDismiss: PropTypes.func.isRequired,
};
