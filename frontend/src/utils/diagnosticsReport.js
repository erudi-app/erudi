// Turning three error sources and two environment summaries into one block of
// text a person can paste into a bug report.
//
// The text is **English on purpose**, in every interface language. It is not
// interface copy: it is machine data read by whoever triages the issue, and a
// report whose field labels change language is harder to read, not easier. The
// chrome around it — headings, buttons, the privacy note — is translated as
// usual.
//
// Nothing here reaches the network. The caller has already read everything
// locally: `/erudi/diagnostics/` over loopback, the app log through the
// preload bridge, and the session buffer from this window's own memory.
//
// This is also the ONE place records are filtered for what merely describes
// the environment (no network, a remote service down, the user's disk or
// port) rather than Erudi itself being wrong (#485's follow-up product
// decision): `isHiddenFromDiagnostics` (utils/bugCounter.js) drops an
// ERROR/CRITICAL entry that matches, so an offline HuggingFace attempt never
// appears in the list, the copied report, OR the sidebar badge -- all three
// read this same merge, so they cannot disagree. WARNINGs are never touched
// by that filter; only the bug-icon badge additionally ignores them (by
// level, in `isCountableError`), not this page. The filter is presentation
// only: `backend.log` and `erudi-backend.log` on disk still carry every
// record, unfiltered.

import { isHiddenFromDiagnostics } from "./bugCounter";

/** Entries kept in the merged timeline. */
export const DEFAULT_ENTRY_LIMIT = 200;

/** The options of the `os` dropdown in .github/ISSUE_TEMPLATE/bug_report.yml. */
const OS_LABELS = {
  "darwin/arm64": "macOS (Apple Silicon)",
  "darwin/x64": "macOS (Intel)",
  win32: "Windows 10 / 11",
  linux: "Linux",
};

/**
 * The bug-report form's OS option for this machine, or null.
 * @param {string} platform - `process.platform`.
 * @param {string} arch - `process.arch`.
 * @returns {string|null} An exact option label from the issue form.
 */
export function osLabel(platform, arch) {
  if (!platform) return null;
  return OS_LABELS[`${platform}/${arch}`] ?? OS_LABELS[platform] ?? null;
}

/** Sortable key; an entry with no timestamp sorts first rather than crashing. */
const timeKey = (entry) => entry?.timestamp ?? "";

/**
 * An app-log record that is the backend's own stdout or stderr, echoed line
 * by line by main.js. The backend writes the same record to `backend.log`,
 * with its continuation lines intact, so when the backend answers the echo
 * is a duplicate of a better copy.
 */
const BACKEND_ECHO_RE = /^Backend (?:stdout|stderr): /;

/** The app-log namespace of the renderer's uncaught-error capture. */
const UNCAUGHT_NAMESPACE = "[renderer:renderer:uncaught]";

/**
 * Merge the backend log, the app log and this window's session errors into one
 * timeline, each error once.
 *
 * Every source is optional. The backend one is the one that goes missing in
 * practice — that is exactly when the other two matter.
 *
 * Two sources overlap by construction, and the overlap is removed here rather
 * than at the writers, because each writer is right to write:
 *
 * - An uncaught renderer error is written to the app log through the
 *   `renderer-log` bridge AND kept in the session buffer. The file record is
 *   the one kept: it survives a reload, and it is what QA reads. The session
 *   entry is dropped when the app log has its record, and only contributes
 *   what the file cannot hold — the repeat count. When the bridge was absent
 *   (a browser, a test, an early boot) the file has nothing and the session
 *   entry stands.
 * - Every backend stdout/stderr line is echoed into the app log by main.js.
 *   When the backend answered, its own log is the source of truth for those
 *   records and the echoes are dropped; when it did not answer, the echoes
 *   are the only copy of its last words and are kept.
 *
 * @param {object} sources - The three sources.
 * @param {object} [sources.backend] - `/erudi/diagnostics/` response, or null.
 * @param {Array} [sources.appLog] - Records from `diagnostics:appLogTail`.
 * @param {Array} [sources.sessionErrors] - Entries from the session buffer.
 * @param {number} [sources.limit] - Maximum entries to keep (newest kept).
 * @returns {Array<{timestamp: string, level: string, source: string, message: string, requestId: ?string, count: number}>} Oldest first.
 */
export function mergeRecentErrors({
  backend = null,
  appLog = [],
  sessionErrors = [],
  limit = DEFAULT_ENTRY_LIMIT,
} = {}) {
  const entries = [];

  for (const record of backend?.recent_errors ?? []) {
    entries.push({
      timestamp: record.timestamp,
      level: record.level,
      source: "backend",
      requestId: record.request_id ?? null,
      message: record.message,
      count: 1,
    });
  }

  const appEntries = [];
  for (const record of appLog ?? []) {
    const message = String(record.message ?? "");
    if (backend && BACKEND_ECHO_RE.test(message)) continue;
    const entry = {
      timestamp: record.timestamp,
      level: record.level,
      source: "app",
      requestId: null,
      message,
      count: 1,
    };
    appEntries.push(entry);
    entries.push(entry);
  }

  for (const record of sessionErrors ?? []) {
    const message = [record.origin, record.message].filter(Boolean).join(": ");
    const count = record.count ?? 1;
    const fileRecord = appEntries.find(
      (entry) => entry.message.startsWith(UNCAUGHT_NAMESPACE) && entry.message.includes(message)
    );
    if (fileRecord) {
      fileRecord.count = Math.max(fileRecord.count, count);
      continue;
    }
    entries.push({
      timestamp: record.timestamp,
      level: "ERROR",
      source: "session",
      requestId: null,
      message,
      count,
      stack: record.stack || undefined,
    });
  }

  // Drop what merely describes the environment before sorting/slicing, so an
  // environmental record never occupies a slot a real error could otherwise
  // hold within `limit`. WARNINGs are untouched -- see the file banner above.
  const visible = entries.filter((entry) => !isHiddenFromDiagnostics(entry));
  visible.sort((a, b) => (timeKey(a) < timeKey(b) ? -1 : timeKey(a) > timeKey(b) ? 1 : 0));
  return limit ? visible.slice(-limit) : visible;
}

/**
 * "16 GB VRAM", not "15.9287109375 GB VRAM": the driver reports full precision.
 *
 * Deliberately NOT `i18n/format.js`'s `formatGigabytes`, which is the right
 * helper everywhere else and the wrong one here. That one is locale aware by
 * design (`Intl.NumberFormat` plus the translated `common:units.gb`), so on a
 * French interface it yields "15,9 Go" — and this string goes into the copied
 * report, which is English in every language on purpose (see the file banner):
 * it is machine data for whoever triages the issue, not interface copy. The
 * panel uses this same helper so the screen and the report cannot drift.
 */
export function formatVram(gigabytes) {
  if (typeof gigabytes !== "number" || !Number.isFinite(gigabytes)) return null;
  return `${Math.round(gigabytes * 10) / 10} GB VRAM`;
}

/** "Apple M3 Pro / Apple M3 Pro GPU, 12 GB VRAM, compute 8.9", or null. */
function hardwareSummary(environment) {
  if (!environment) return null;
  const gpu = [
    environment.gpu_name,
    environment.vram_total_gb ? formatVram(environment.vram_total_gb) : null,
    environment.compute_capability ? `compute ${environment.compute_capability}` : null,
  ]
    .filter(Boolean)
    .join(", ");
  const parts = [environment.cpu_model, gpu || null].filter(Boolean);
  return parts.length ? parts.join(" / ") : null;
}

/**
 * Values for the bug-report form's fields.
 *
 * The version and the OS come from Electron, so they survive a dead backend —
 * which is precisely the report we most want to receive.
 *
 * `hardware` may be given explicitly, for a caller that knows the machine
 * better than the backend does: the engine-failure dialog is handed the card
 * and its compute capability by the notice that raised it, and opens when the
 * backend either cannot describe that card or is not answering at all.
 *
 * @param {object} sources - Where the values come from.
 * @param {object} [sources.app] - `app:getInfo` result.
 * @param {object} [sources.backend] - `/erudi/diagnostics/` response, or null.
 * @param {string} [sources.hardware] - Overrides the backend's hardware line.
 * @returns {{version: ?string, os: ?string, hardware: ?string, model: ?string}} Field values.
 */
export function buildPrefill({ app = null, backend = null, hardware = null } = {}) {
  const environment = backend?.environment ?? null;
  return {
    version: app?.version ?? null,
    os: osLabel(app?.platform, app?.arch),
    hardware: hardware || hardwareSummary(environment),
    model: environment?.loaded_model ?? null,
  };
}

/** One "Label: value" line, or nothing when there is no value. */
function line(label, value) {
  return value === null || value === undefined || value === "" ? null : `${label}: ${value}`;
}

/**
 * Format the whole report as plain text.
 *
 * @param {object} sources - What to describe.
 * @param {object} [sources.app] - `app:getInfo` result.
 * @param {object} [sources.backend] - `/erudi/diagnostics/` response, or null.
 * @param {Array} [sources.entries] - Merged timeline from mergeRecentErrors.
 * @returns {string} The text the user copies.
 */
export function formatDiagnosticsText({ app = null, backend = null, entries = [] } = {}) {
  const environment = backend?.environment ?? null;
  const lines = [
    `Erudi ${app?.version ?? "unknown version"}`,
    line("OS", osLabel(app?.platform, app?.arch) ?? app?.platform),
    line(
      "OS detail",
      environment
        ? [environment.platform, environment.platform_release, environment.architecture]
            .filter(Boolean)
            .join(" ")
        : null
    ),
    line("Electron", app?.electron),
  ];

  if (environment) {
    lines.push(
      line("Engine", environment.engine),
      line("Hardware", hardwareSummary(environment)),
      line("Model in memory", environment.loaded_model ?? environment.loaded_model_id),
      line("Backend Python", environment.python_version),
      line("Database", environment.db),
      line("Backend log", environment.backend_log_path)
    );
  } else {
    // Say what is missing. A report that silently omits the backend reads as
    // "there was nothing wrong there", which is the opposite of the truth.
    lines.push("Backend: the backend did not answer, so its part of this report is missing.");
  }

  lines.push(line("App log", app?.appLogPath));
  lines.push("", "Recent errors (WARNING and above, oldest first):");

  if (!entries.length) {
    lines.push("  (no warnings or errors recorded)");
  } else {
    for (const entry of entries) {
      const repeat = entry.count > 1 ? ` x${entry.count}` : "";
      const rid = entry.requestId ? ` [${entry.requestId}]` : "";
      const head = `  ${entry.timestamp} [${entry.level}] (${entry.source})${rid}${repeat}`;
      lines.push(`${head} ${String(entry.message ?? "").replace(/\n/g, "\n    ")}`);
      if (entry.stack) lines.push(`    ${entry.stack.replace(/\n/g, "\n    ")}`);
    }
  }

  return lines.filter((value) => value !== null).join("\n");
}
