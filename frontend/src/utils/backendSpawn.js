// Spawn options for the backend child process (#142).
//
// The Windows backend is a console-subsystem PyInstaller exe (backend.spec:
// console=True, kept so Electron can read the stdout JSON lifecycle events).
// Without windowsHide, Windows allocates a console window for it — an empty
// window that flashes on launch and again briefly whenever the backend crashes.
// stdout still flows through the piped stdio, so hiding the window costs nothing
// (a visible console is NOT required to capture stdout).
//
// `detached` is POSIX-only: killBackend tears the whole tree down with
// kill(-pid) there, but on Windows it uses `taskkill /F /T`, so detaching is
// unnecessary and only contributes the console window.

// Variables the backend must never inherit. `langsmith` rides into the frozen
// build as a transitive dependency of langchain, and ANY of these is enough to
// turn on cloud tracing: LangChain would then POST the system prompt, the
// knowledge-base excerpts, the user's question and the model's answer to
// api.smith.langchain.com, with nothing in the UI to say so. Someone who
// exported LANGCHAIN_TRACING_V2 years ago for an unrelated project must not
// have their conversations exfiltrated by us. The backend also forces the two
// tracing flags off itself (backend/run.py); stripping the whole family here is
// the second lock, and it covers names we have not thought of.
const BLOCKED_ENV_PREFIXES = ["LANGCHAIN_", "LANGSMITH_"];

/**
 * Build the environment for the backend child process.
 *
 * Starts from the parent environment — the backend legitimately needs PATH,
 * HOME, HF_TOKEN and the rest — then removes what must not cross, and adds the
 * launcher's own variables.
 *
 * PYTHONUTF8=1 only affects the NON-frozen (dev) interpreter: PyInstaller's
 * bootloader pre-initializes CPython and ignores it, so the packaged build gets
 * UTF-8 mode from the spec's interpreter OPTIONS instead (backend/backend.spec,
 * #168). It is still set here because in dev it makes open() read bundled data
 * files (e.g. alembic.ini) as UTF-8 whatever the locale — a macOS app launched
 * from Finder inherits no LANG (#149).
 *
 * ERUDI_WATCH_STDIN=1 opts the launcher into watching stdin for EOF: on quit we
 * close stdin so the backend shuts down gracefully (stop_postgres), which
 * Windows otherwise never got because taskkill /F /T skips the lifespan (#216).
 *
 * ERUDI_APP_VERSION is the app version (`app.getVersion()`): the backend has
 * none of its own, and gives backend.log to the running version with it, so the
 * Diagnostics page lists that version's records only (#576). Only the value
 * main passes is used; one inherited from the parent environment is dropped.
 *
 * @param {object} parentEnv - typically process.env; never mutated
 * @param {{appVersion?: string}} [options] - the app version to pass
 * @returns {object} the environment to hand to spawn
 */
export function buildBackendEnv(parentEnv, { appVersion } = {}) {
  const env = {};
  for (const [key, value] of Object.entries(parentEnv)) {
    if (BLOCKED_ENV_PREFIXES.some((prefix) => key.startsWith(prefix))) continue;
    if (key === "ERUDI_APP_VERSION") continue;
    env[key] = value;
  }
  env.PYTHONUTF8 = "1";
  env.ERUDI_WATCH_STDIN = "1";
  if (appVersion) env.ERUDI_APP_VERSION = appVersion;
  return env;
}

/**
 * Build the child_process.spawn options for the backend.
 * @param {string} platform - process.platform ("win32" | "darwin" | "linux")
 * @param {{ cwd: string, env: object }} io - working dir + environment
 * @returns {object} spawn options
 */
export function buildBackendSpawnOptions(platform, { cwd, env }) {
  const isWin = platform === "win32";
  return {
    stdio: ["pipe", "pipe", "pipe"],
    cwd,
    env,
    // POSIX: own process group so killBackend can kill(-pid) the tree.
    detached: !isWin,
    // Windows: suppress the console-subsystem window; no-op elsewhere.
    windowsHide: true,
  };
}
