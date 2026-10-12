import { formatPercent } from "../i18n/format";

/**
 * The compaction thresholds a conversation (or the global default) can carry:
 * the share of the model's working context window, in whole percent, at which
 * older messages are summarized (#607). Mirrors the backend scale
 * (`src/engines/working_window.py:COMPACTION_THRESHOLDS`, 50 to 90 in steps
 * of 5) so the chat header, the pre-conversation panel and the Settings page
 * offer the same `<select>` options.
 */
export const COMPACTION_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 90];

export const DEFAULT_COMPACTION_THRESHOLD = 80;

/** "80%" / "80 %": an option label in the active language. */
export function formatCompactionThreshold(value) {
  return formatPercent(value, { maximumFractionDigits: 0 });
}
