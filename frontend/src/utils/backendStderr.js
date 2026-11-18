// Interpreting the backend's stderr — for LOGGING only, never as a fatal signal.
//
// The backend (`backend/run.py`) emits authoritative `startup_error` JSON events
// on stdout for genuine failures, and a process exit is caught separately. So a
// stderr *substring* must never be turned into a fatal error: on a CPU build
// (no NVIDIA GPU) normal startup prints lines like "CUDA not available",
// "pynvml ... NVML", and SQLAlchemy/psycopg log lines containing "database" —
// the old heuristic flagged those as fatal and aborted the UI on a healthy
// machine. classifyStderrLine returns a non-fatal diagnostic hint (or null);
// isBenignStderrLine says whether a line is expected noise on a CPU build.

// Substrings that are BENIGN on a CPU build and appear during normal startup.
const BENIGN_ON_CPU = [
  "CUDA",
  "cuda:",
  "pynvml",
  "NVML",
  "mlx.core",
  "MLX",
  "database",
  "psycopg",
  "sqlalchemy",
];

/** True when the stderr line is expected noise on a CPU build (never fatal). */
export function isBenignStderrLine(line) {
  if (!line) return false;
  return BENIGN_ON_CPU.some((p) => line.includes(p));
}

/**
 * Classify a stderr line for LOGGING/diagnostics only. Always non-fatal — the
 * returned object never carries `fatal: true`. Returns null for unremarkable
 * lines. The caller must NOT emit a startup_error from this; readiness/failure
 * is decided by the backend's own `startup_error` JSON events and process exit.
 */
export function classifyStderrLine(line) {
  if (!line) return null;
  if (line.includes("No module named")) {
    const match = line.match(/No module named '([^']+)'/);
    const mod = match ? match[1] : "unknown";
    return { code: "MISSING_DEPENDENCY", message: `Missing Python module: ${mod}`, fatal: false };
  }
  if (isBenignStderrLine(line)) {
    return { code: "GPU_DIAGNOSTIC", message: line, fatal: false };
  }
  return null;
}
