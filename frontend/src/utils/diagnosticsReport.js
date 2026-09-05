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
 * Merge the backend log, the app log and this window's session errors into one
 * timeline.
 *
 * Every source is optional. The backend one is the one that goes missing in
 * practice — that is exactly when the other two matter.
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

  for (const record of appLog ?? []) {
    entries.push({
      timestamp: record.timestamp,
      level: record.level,
      source: "app",
      requestId: null,
      message: record.message,
      count: 1,
    });
  }

  for (const record of sessionErrors ?? []) {
    entries.push({
      timestamp: record.timestamp,
      level: "ERROR",
      source: "session",
      requestId: null,
      message: [record.origin, record.message].filter(Boolean).join(": "),
      count: record.count ?? 1,
      stack: record.stack || undefined,
    });
  }

  entries.sort((a, b) => (timeKey(a) < timeKey(b) ? -1 : timeKey(a) > timeKey(b) ? 1 : 0));
  return limit ? entries.slice(-limit) : entries;
}

/** "Apple M3 Pro / Apple M3 Pro GPU, 12 GB VRAM, compute 8.9", or null. */
function hardwareSummary(environment) {
  if (!environment) return null;
  const gpu = [
    environment.gpu_name,
    environment.vram_total_gb ? `${environment.vram_total_gb} GB VRAM` : null,
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
 * @param {{app?: object, backend?: object}} sources - App info and backend response.
 * @returns {{version: ?string, os: ?string, hardware: ?string, model: ?string}} Field values.
 */
export function buildPrefill({ app = null, backend = null } = {}) {
  const environment = backend?.environment ?? null;
  return {
    version: app?.version ?? null,
    os: osLabel(app?.platform, app?.arch),
    hardware: hardwareSummary(environment),
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
