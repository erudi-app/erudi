import React, { useCallback, useState } from "react";
import PropTypes from "prop-types";
import { Check, Copy, Github } from "lucide-react";
import { useTranslation } from "react-i18next";

import { createLogger } from "../utils/logger";

const log = createLogger("ReportProblem");

/** The bug report form. `template` selects it in the issue chooser. */
export const ISSUE_FORM_URL = "https://github.com/erudi-app/erudi/issues/new";
export const ISSUE_FORM_TEMPLATE = "bug_report.yml";

/** For people without a GitHub account. */
export const CONTACT_PAGE_URL = "https://erudi.app/contact";

/** How long the "Copied" confirmation stays up. */
const COPIED_MS = 2000;

/**
 * Build the prefilled bug-report URL.
 *
 * The parameter names are the `id` of each field in
 * `.github/ISSUE_TEMPLATE/bug_report.yml` — that is what GitHub matches on.
 * `version`, `hardware` and `model` are text fields, which GitHub's
 * documentation says can be prefilled. `os` is a dropdown; prefilling those is
 * not documented, so it is sent best-effort (an unrecognised parameter is
 * ignored) and the same value is written into the diagnostics text, which the
 * user pastes regardless.
 *
 * The `logs` field is deliberately absent. A log tail is thousands of
 * characters and GitHub answers 414 to a URL that long, so logs travel by
 * clipboard: the button copies first, then opens the form.
 *
 * @param {{version?: string, os?: string, hardware?: string, model?: string}} prefill - Field values.
 * @returns {string} Absolute URL to the prefilled form.
 */
export function buildIssueUrl(prefill = {}) {
  const params = new URLSearchParams({ template: ISSUE_FORM_TEMPLATE });
  for (const key of ["version", "os", "hardware", "model"]) {
    const value = prefill?.[key];
    if (value) params.set(key, String(value));
  }
  return `${ISSUE_FORM_URL}?${params.toString()}`;
}

/**
 * The canonical "here is what to send us" block: a copy route to the
 * diagnostics text, the GitHub form and the contact-page fallback.
 *
 * Reused wherever the app has to hand a problem back to the user — the
 * Diagnostics page, the error-boundary screen, and any modal that wants to
 * offer the same route out.
 *
 * Two renderings share this one component rather than forking into two,
 * because the copy handler, the issue-form handler and the propTypes would
 * otherwise have to be kept in sync by hand:
 * - `preview` (default `true`) is the original block: a read-only textarea
 *   holding the full text, the optional caller `note`, a fixed instruction
 *   paragraph, and a plain "Copy" button. `EngineFailureModal` and
 *   `AppErrorBoundary` use this rendering, unchanged.
 * - `preview={false}` is the Diagnostics page's lighter variant: no textarea,
 *   no note, no long instruction. A single "Copy the full report" button (and
 *   a one-line paste hint) appears only when the caller passes `hasErrors`,
 *   because there is nothing to hand a maintainer when nothing was recorded.
 *   The GitHub button and the contact sentence stay in both states.
 *
 * @param {object} props - Component props.
 * @param {string} props.diagnostics - Plain text the user copies and pastes.
 * @param {object} [props.prefill] - Values for the issue form's text fields.
 * @param {string} [props.note] - A caution shown under the text in preview
 *   mode, when what it holds deserves one. Ignored when `preview` is false.
 * @param {boolean} [props.preview] - Render the full text preview (textarea,
 *   note, long instruction). Defaults to `true`.
 * @param {boolean} [props.hasErrors] - In `preview={false}` mode only: whether
 *   there is something to report, which gates the copy button and the short
 *   paste hint. Ignored in preview mode.
 * @returns {JSX.Element} The block.
 */
export default function ReportProblem({
  diagnostics,
  prefill,
  note,
  preview = true,
  hasErrors = false,
}) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(diagnostics);
      setCopied(true);
      setTimeout(() => setCopied(false), COPIED_MS);
      return true;
    } catch (error) {
      // A denied clipboard must not block the report: the text is on screen
      // and selectable either way.
      log.warn("Could not copy the diagnostics to the clipboard", error);
      return false;
    }
  }, [diagnostics]);

  const openIssueForm = useCallback(async () => {
    await copy();
    // main.js routes http(s) window.open to the system browser.
    window.open(buildIssueUrl(prefill), "_blank");
  }, [copy, prefill]);

  // In preview mode the copy button is always offered, next to the text it
  // duplicates. In the lighter mode it is the only way to grab the report, so
  // it only earns its place when there is something worth sending.
  const showCopyButton = preview || hasErrors;

  return (
    <div className="space-y-3">
      <h3 className="text-[13px] font-semibold text-[var(--ink)]">
        {t("diagnostics:report.heading")}
      </h3>

      {preview && (
        <textarea
          aria-label={t("diagnostics:report.textareaLabel")}
          readOnly
          value={diagnostics}
          rows={8}
          className="w-full font-mono text-[11px] leading-relaxed rounded-lg border border-[var(--line)] bg-[var(--canvas)] text-[var(--ink-dim)] p-3 custom-scroll resize-y"
        />
      )}

      {preview && note && (
        <p className="text-[12px] text-[var(--ink-faint)] leading-relaxed">{note}</p>
      )}

      {preview && (
        <p className="text-[12px] text-[var(--ink-faint)] leading-relaxed">
          {t("diagnostics:report.instruction")}
        </p>
      )}

      {!preview && hasErrors && (
        <p className="text-[12px] text-[var(--ink-faint)] leading-relaxed">
          {t("diagnostics:report.pasteHint")}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {showCopyButton && (
          <button
            type="button"
            onClick={copy}
            className="inline-flex items-center gap-1.5 text-[13px] rounded-lg border border-[var(--line)] bg-[var(--canvas)] text-[var(--ink)] px-3 py-1.5 hover:border-[var(--fit-good)] focus:outline-none focus:border-[var(--fit-good)] transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            {copied
              ? t("common:actions.copied")
              : preview
                ? t("common:actions.copy")
                : t("diagnostics:report.copyFullReport")}
          </button>
        )}

        <button
          type="button"
          onClick={openIssueForm}
          className="inline-flex items-center gap-1.5 text-[13px] rounded-lg border border-[var(--line)] bg-[var(--canvas)] text-[var(--ink)] px-3 py-1.5 hover:border-[var(--fit-good)] focus:outline-none focus:border-[var(--fit-good)] transition-colors"
        >
          <Github className="w-3.5 h-3.5" />
          {t("diagnostics:report.github")}
        </button>
      </div>

      <p className="text-[12px] text-[var(--ink-faint)] leading-relaxed">
        {t("diagnostics:report.contactPrefix")}{" "}
        <a
          href={CONTACT_PAGE_URL}
          target="_blank"
          rel="noreferrer"
          className="text-[var(--fit-good)] hover:underline"
        >
          {t("diagnostics:report.contactLink")}
        </a>{" "}
        {t("diagnostics:report.contactSuffix")}
      </p>
    </div>
  );
}

ReportProblem.propTypes = {
  diagnostics: PropTypes.string.isRequired,
  prefill: PropTypes.shape({
    version: PropTypes.string,
    os: PropTypes.string,
    hardware: PropTypes.string,
    model: PropTypes.string,
  }),
  note: PropTypes.string,
  preview: PropTypes.bool,
  hasErrors: PropTypes.bool,
};
