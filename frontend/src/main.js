/* eslint-disable no-console */
const { app, BrowserWindow, ipcMain, dialog, Menu, shell } = require("electron");
const { createMainTranslator } = require("./i18n/mainTranslator");

// Interface language for the native menu and dialogs (#385). The renderer
// resolves the language (local mirror, OS locale, then the persisted user
// setting) and pushes it over IPC; main only mirrors it and rebuilds the menu.
const i18n = createMainTranslator("en");
const t = (key, vars) => i18n.t(key, vars);
const path = require("node:path");
const { spawn, execSync } = require("child_process");
const fs = require("fs");
const os = require("os");

// Pure, unit-tested startup helpers (shared with the renderer).
const { confirmBackendHealth } = require("./utils/backendHealth");
const { classifyStderrLine } = require("./utils/backendStderr");
const { shouldRetrySpawn } = require("./utils/backendRetry");
const { buildBackendSpawnOptions, buildBackendEnv } = require("./utils/backendSpawn");
const { gracefulShutdown } = require("./utils/backendShutdown");
const { readAppLogTail } = require("./utils/appLogTail");
const { describeProcessGone, formatMainRecord } = require("./utils/mainLog");
const { createBackendStartTracker, requestStop } = require("./utils/backendStartTracker");

// electron-updater: only loaded in production to avoid dev noise.
// Reads latest.yml / latest-mac.yml from GitHub Releases and handles
// download + install of new versions.
let autoUpdater = null;
// Kept until the log file is set up below, then logged: electron-updater ships
// with every packaged build, so a require that fails is a packaging defect.
let updaterLoadError = null;
if (app.isPackaged) {
  try {
    autoUpdater = require("electron-updater").autoUpdater;
    autoUpdater.logger = require("electron-log");
    autoUpdater.logger.transports.file.level = "info";
  } catch (e) {
    updaterLoadError = e;
    autoUpdater = null;
  }
}

// Automatic updates are refusable in Settings. The preference is persisted with
// the other user settings, which only the renderer can read, so main starts out
// not knowing it and does NOT check for updates until the renderer says so:
// checking early would send a request the user may have declined. Enabled is
// what the renderer reports when the setting cannot be read, so the shipped
// behaviour is unchanged for anyone who never opens the setting.
let autoUpdateEnabled = null;
let updatePollTimer = null;

// Renderer + preload entry resolution (replaces the @electron-forge/plugin-webpack
// magic globals). Prod (packaged): load the built files that sit next to this
// main bundle — .webpack/renderer/main_window/ — via file://. Dev: the
// webpack-dev-server serves the renderer at :3000 and writes preload.js to disk.
const MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY = path.join(
  __dirname,
  "..",
  "renderer",
  "main_window",
  "preload.js"
);
const MAIN_WINDOW_RENDERER_INDEX = path.join(
  __dirname,
  "..",
  "renderer",
  "main_window",
  "index.html"
);
const RENDERER_DEV_URL = "http://localhost:3000/";

let backendProcess = null;
// Whether main asked a child to stop is recorded ON that child
// (requestStop, utils/backendStartTracker.js): a restart whose graceful
// shutdown times out spawns the replacement before the old child's exit
// event arrives, and a global flag reset at spawn would call that a crash.
let mainWindow = null;
let isCreatingWindow = false;
// Set once a quit-time graceful shutdown is in flight so before-quit runs its
// (async) teardown exactly once — the app.quit() it re-issues must fall through.
let shuttingDown = false;
// Backend readiness state, published to the renderer (covers the race where
// readiness happens before the renderer attaches its event listener — the
// renderer queries `backend:getInfo` on mount).
let resolvedPort = null;
let backendIsReady = false;
// Transient failures (port contention) may auto-respawn; deterministic ones
// fail fast and wait for a manual retry.
const MAX_SPAWN_ATTEMPTS = 2;

// Kill the backend process and its entire child tree.
// On Windows, SIGTERM only kills the parent — llama-server.exe is left orphaned.
// taskkill /F /T kills the full process tree.
function killBackend(proc) {
  if (!proc) {
    return;
  }
  if (process.platform === "win32") {
    try {
      execSync(`taskkill /F /T /PID ${proc.pid}`, { stdio: "ignore" });
    } catch (_) {
      // Process may have already exited
    }
  } else {
    try {
      // Kill the entire process group (negative PID) so uvicorn workers
      // and multiprocessing children are also terminated.
      process.kill(-proc.pid, "SIGTERM");
    } catch (_) {
      // Fallback if the process is no longer a group leader
      try {
        proc.kill("SIGTERM");
      } catch (_) {
        /* already exited */
      }
    }
  }
}

// Create a log file for debugging. This is THE file QA reads (app:getLogPath);
// renderer logs are forwarded here too via the "renderer-log" IPC channel.
const logFile = path.join(os.tmpdir(), "erudi-backend.log");
const oldLogFile = path.join(os.tmpdir(), "erudi-backend.old.log");
const LOG_MAX_BYTES = 10 * 1024 * 1024; // 10 MB size cap
const LOG_STAT_EVERY = 200; // fs.stat is cheap but not free — sample it
let logWriteCount = 0;

// Size-cap rotation: stat the file on the first write and then every
// LOG_STAT_EVERY writes; past the cap, the current file replaces
// erudi-backend.old.log and a fresh one starts. Never throws.
const rotateLogIfNeeded = () => {
  logWriteCount += 1;
  if (logWriteCount % LOG_STAT_EVERY !== 1) return;
  try {
    const { size } = fs.statSync(logFile);
    if (size <= LOG_MAX_BYTES) return;
    fs.rmSync(oldLogFile, { force: true }); // rename() won't replace on Windows
    fs.renameSync(logFile, oldLogFile);
  } catch (_) {
    // Missing file or a losing race — never let rotation break logging.
  }
};

const log = (message) => {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${message}\n`;
  console.log(message);
  try {
    rotateLogIfNeeded();
    fs.appendFileSync(logFile, logMessage);
  } catch (err) {
    console.error("Failed to write to log file:", err);
  }
};

// Levelled records for main's own failures. `log()` writes unlevelled lines,
// which the Diagnostics panel skips on purpose (utils/appLogTail.js); these two
// state the level in the same shape the renderer bridge uses, so a backend that
// died or a renderer that crashed reaches the page that exists to report bugs.
const logWarn = (message, error) => log(formatMainRecord("WARN", message, error));
const logError = (message, error) => log(formatMainRecord("ERROR", message, error));

log(`Starting app, log file: ${logFile}`);
if (updaterLoadError) {
  logWarn("electron-updater could not be loaded; automatic updates are off", updaterLoadError);
}

// ── What no catch sees ────────────────────────────────────────────────────────
// Each handler writes one ERROR record and changes nothing else about how the
// app behaves: `uncaughtExceptionMonitor` observes without replacing Electron's
// default handling of an uncaught exception (its dialog stays); an unhandled
// rejection in the main process is otherwise a warning on stderr after which
// the process goes on, and it still does; a process that is gone is gone.
process.on("uncaughtExceptionMonitor", (error, origin) => {
  logError(`Uncaught exception in the main process (${origin})`, error);
});
process.on("unhandledRejection", (reason) => {
  logError("Unhandled promise rejection in the main process", reason);
});
app.on("child-process-gone", (_event, details) => {
  logError(describeProcessGone(details?.type || "child", details));
});
app.on("render-process-gone", (_event, _contents, details) => {
  logError(describeProcessGone("renderer", details));
});

if (require("electron-squirrel-startup")) {
  app.quit();
}

function resolvePackagedBackendPath() {
  // On Windows PyInstaller produces backend.exe; on macOS/Linux just backend
  const exeSuffix = process.platform === "win32" ? ".exe" : "";
  // Candidate locations inside the packaged app
  const candidates = [
    path.join(process.resourcesPath, "backend", `backend${exeSuffix}`),
    path.join(process.resourcesPath, "backend", "backend", `backend${exeSuffix}`),
    path.join(process.resourcesPath, "app.asar.unpacked", "backend", `backend${exeSuffix}`),
  ];

  log("Checking packaged backend paths...");
  log(`process.resourcesPath: ${process.resourcesPath}`);

  for (const c of candidates) {
    log(`Checking candidate: ${c}`);
    try {
      if (fs.existsSync(c)) {
        const stat = fs.statSync(c);
        log(`Found ${c}, isFile: ${stat.isFile()}, isExecutable: ${!!(stat.mode & 0o111)}`);
        if (stat.isFile()) {
          return c;
        }
      } else {
        log(`Path does not exist: ${c}`);
      }
    } catch (error) {
      logWarn(`Could not check the backend candidate ${c}`, error);
    }
  }
  return null;
}

const startRealBackend = () => {
  return new Promise((resolve, reject) => {
    log("Starting backend server...");

    // In development mode, assume backend is already running via dev-start.sh
    if (!app.isPackaged) {
      log("Development mode: assuming backend is running via dev-start.sh");
      // Just check if backend is responding
      const checkDevBackendHealth = async () => {
        const devPort = process.env.BACKEND_PORT || "27182";
        log(`Dev mode using port: ${devPort}`);

        for (let i = 0; i < 10; i++) {
          try {
            log(`Dev backend health check attempt ${i + 1}/10`);
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 2000);

            const response = await fetch(`http://127.0.0.1:${devPort}/erudi/health/`, {
              signal: controller.signal,
            });
            clearTimeout(timeoutId);

            if (response.ok) {
              const data = await response.json();
              log(`Backend is ready: ${data.message}`);
              resolve({ port: Number(devPort) });
              return;
            }
          } catch (error) {
            log(`Dev backend health check failed: ${error.message}`);
          }

          // Wait before next attempt
          await new Promise((r) => setTimeout(r, 1000));
        }

        // If we get here, backend is not responding
        logError(
          `Backend is not responding on localhost:${devPort}. Make sure to run: scripts/dev/dev-start.sh or set BACKEND_PORT env variable`
        );
        reject(new Error(`Backend is not responding on localhost:${devPort}`));
      };

      checkDevBackendHealth();
      return;
    }

    // Production mode: spawn packaged backend.
    // 27182 = Erudi's canonical port (digits of e, for erudites — see backend/run.py).
    const PORT = 27182;

    // The backend owns port selection: it scans 27182–27199 and announces the
    // resolved port back to us via its JSON lifecycle events (we forward it to the
    // renderer), so it's fine if this exact port is taken.
    let backendPath;
    if (app.isPackaged) {
      backendPath = resolvePackagedBackendPath();
    } else {
      const exeSuffix = process.platform === "win32" ? ".exe" : "";
      const devCandidates = [
        path.join(__dirname, "..", "..", "backend", "dist", "backend", `backend${exeSuffix}`),
        path.join(__dirname, "..", "..", "backend", "backend"),
      ];
      backendPath = devCandidates.find((p) => fs.existsSync(p)) || null;
    }

    if (!backendPath || !fs.existsSync(backendPath)) {
      const error =
        `Backend executable not found. Checked path: ${backendPath || "None"}\n` +
        "You likely need to build it first (e.g. 'pyinstaller backend.spec').";
      logError(error);
      reject(new Error(error));
      return;
    }

    log(`Spawning backend: ${backendPath} --port ${PORT}`);
    const workingDir = path.dirname(backendPath);
    log(`Working directory: ${workingDir}`);

    // Storage paths (embedded PostgreSQL data dir, model cache) are resolved
    // by the backend itself (src/launcher/runtime_paths.py) — nothing to pass.
    // buildBackendEnv inherits the parent environment, strips what must never
    // cross into the backend (the LangChain/LangSmith family, which would turn
    // on cloud tracing of every conversation), and adds PYTHONUTF8 and
    // ERUDI_WATCH_STDIN. See utils/backendSpawn.js for the reasoning.
    const backendEnv = buildBackendEnv(process.env);

    backendProcess = spawn(
      backendPath,
      ["--port", PORT.toString()],
      buildBackendSpawnOptions(process.platform, { cwd: workingDir, env: backendEnv })
    );

    log(`Backend process spawned with PID: ${backendProcess.pid}`);

    const proc = backendProcess;
    let actualPort = PORT;
    let capTimer = null;
    // The tracker is the single owner of the start-failure record: the first
    // cause to arrive (startup_error, spawn error, exit, safety cap) writes
    // the one ERROR and settles the promise; everything after it is a plain
    // line. reject carries the error CODE (string) so the supervisor can
    // classify it.
    const tracker = createBackendStartTracker({
      log,
      logError,
      onFail: (code) => {
        if (capTimer) clearTimeout(capTimer);
        reject(new Error(code));
      },
      onSucceed: () => {
        if (capTimer) clearTimeout(capTimer);
        resolve({ port: actualPort });
      },
    });
    const succeed = tracker.succeed;
    const failWith = tracker.fail;

    // Absolute safety cap. The backend self-aborts at its own first-run-aware
    // budget (300s first run / 120s after) and emits startup_error, which we
    // catch below; this only fires if it goes completely silent. We NEVER kill
    // the backend just for being slow — that was the 30s-kill bug.
    const MAX_READY_WAIT_MS = 330000;
    capTimer = setTimeout(() => {
      failWith(
        "PORT_TIMEOUT",
        `did not report ready within the ${MAX_READY_WAIT_MS / 1000}s safety cap`
      );
    }, MAX_READY_WAIT_MS);

    backendProcess.stdout.on("data", (data) => {
      for (const line of data.toString().split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        log(`Backend stdout: ${trimmed}`);
        let event = null;
        try {
          event = JSON.parse(trimmed);
        } catch (_) {
          continue; // ordinary (non-JSON) log line
        }
        if (!event || !event.event) continue;
        // Forward every structured event to the renderer (starting/phase/ready/…).
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send("backend-event", event);
        }
        if (event.event === "starting" && event.port) {
          actualPort = event.port;
          log(`Backend selected port: ${actualPort}`);
        } else if (event.event === "startup_error") {
          // The cause, with its traceback, is in backend.log (or on the
          // backend's stderr, echoed above); this is the parent's record.
          failWith(
            event.code || "BACKEND_STARTUP_FAILED",
            `backend reported startup_error${event.message ? `: ${event.message}` : ""}`
          );
        } else if (event.event === "ready") {
          if (event.port) actualPort = event.port;
          log(`Backend reported ready on port ${actualPort}; confirming health...`);
          confirmBackendHealth({
            fetchFn: (url) => fetch(url),
            url: `http://127.0.0.1:${actualPort}/erudi/health/`,
          })
            .then((ok) => {
              if (ok) {
                log("Backend health confirmed.");
                succeed();
              } else {
                failWith(
                  "BACKEND_UNREACHABLE",
                  `reported ready on port ${actualPort} but health could not be confirmed`
                );
              }
            })
            .catch((error) => {
              failWith(
                "BACKEND_UNREACHABLE",
                `health check of port ${actualPort} threw: ${error.message}`,
                error
              );
            });
        }
      }
    });

    backendProcess.stderr.on("data", (data) => {
      const output = data.toString().trim();
      if (!output) return;
      log(`Backend stderr: ${output}`);
      // stderr is advisory only. The backend emits authoritative startup_error
      // events on stdout; a stderr substring must never be treated as fatal — a
      // CPU build prints benign lines like "CUDA not available" / NVML / SQLAlchemy
      // "database" logs during a perfectly healthy boot. Log a hint at most.
      const hint = classifyStderrLine(output);
      if (hint) {
        // A missing Python module is a packaging defect; the GPU diagnostics
        // are the expected noise of a CPU build and stay unlevelled.
        const write = hint.code === "MISSING_DEPENDENCY" ? logError : log;
        write(`stderr hint: ${hint.code} - ${hint.message}`);
      }
    });

    backendProcess.on("exit", (code, signal) => {
      // The tracker decides the level: a stop main asked for, the crash that
      // is the start failure, a running backend that died, or the exit that
      // follows an already-recorded failure.
      tracker.exited(proc, code, signal);
      if (backendProcess === proc) backendProcess = null;
    });

    backendProcess.on("error", (error) => {
      failWith(
        "BACKEND_SPAWN_FAILED",
        `could not start the backend process ${backendPath}: ${error.message}`,
        error
      );
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("backend-event", {
          event: "startup_error",
          code: "BACKEND_SPAWN_FAILED",
          message: error.message,
          source: "spawn",
        });
      }
    });
  });
};

// Supervise a backend spawn: auto-respawn only transient failures (port
// contention), fail fast + surface deterministic ones for a manual retry.
async function startBackendSupervised(attempt = 0) {
  try {
    log(`Backend startup attempt ${attempt + 1}...`);
    const { port } = await startRealBackend();
    resolvedPort = port;
    backendIsReady = true;
    log(`Backend is ready on port ${port}.`);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("backend-event", { event: "backend_ready", port });
    }
  } catch (error) {
    const code = (error && error.message) || "BACKEND_STARTUP_FAILED";
    log(`Backend start attempt ${attempt + 1} failed: ${code}`);
    if (shouldRetrySpawn(code, attempt, MAX_SPAWN_ATTEMPTS)) {
      logWarn(`Transient backend start failure (${code}); respawning`);
      requestStop(backendProcess);
      killBackend(backendProcess);
      backendProcess = null;
      await new Promise((r) => setTimeout(r, 2000));
      return startBackendSupervised(attempt + 1);
    }
    // The failure itself is already recorded (one ERROR, by startRealBackend's
    // tracker, which knows the cause); this line only says it reached the user.
    log(`Backend startup failed (${code}); surfacing to the user`);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("backend-event", {
        event: "startup_error",
        code,
        message: (error && error.message) || code,
        source: "startup",
      });
    }
  }
}

// Create application menu with Help options
const createApplicationMenu = () => {
  const isMac = process.platform === "darwin";

  const template = [
    // App menu (macOS only)
    ...(isMac
      ? [
          {
            label: app.name,
            submenu: [
              { role: "about" },
              { type: "separator" },
              { role: "services" },
              { type: "separator" },
              { role: "hide" },
              { role: "hideOthers" },
              { role: "unhide" },
              { type: "separator" },
              { role: "quit" },
            ],
          },
        ]
      : []),

    // File menu
    {
      label: t("menu.file"),
      submenu: [isMac ? { role: "close" } : { role: "quit" }],
    },

    // Edit menu
    {
      label: t("menu.edit"),
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        ...(isMac
          ? [
              { role: "pasteAndMatchStyle" },
              { role: "delete" },
              { role: "selectAll" },
              { type: "separator" },
              {
                label: t("menu.speech"),
                submenu: [{ role: "startSpeaking" }, { role: "stopSpeaking" }],
              },
            ]
          : [{ role: "delete" }, { type: "separator" }, { role: "selectAll" }]),
      ],
    },

    // View menu
    {
      label: t("menu.view"),
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },

    // Window menu
    {
      label: t("menu.window"),
      submenu: [
        { role: "minimize" },
        { role: "zoom" },
        ...(isMac
          ? [{ type: "separator" }, { role: "front" }, { type: "separator" }, { role: "window" }]
          : [{ role: "close" }]),
      ],
    },

    // Help menu
    {
      role: "help",
      submenu: [
        {
          label: t("menu.help.openDataFolder"),
          click: async () => {
            try {
              const dataDir = getDataDirectory();

              // Create directory if it doesn't exist
              if (!fs.existsSync(dataDir)) {
                fs.mkdirSync(dataDir, { recursive: true });
              }

              // Open in Finder
              shell.openPath(dataDir);
              log(`Opened data folder: ${dataDir}`);
            } catch (error) {
              logError("Failed to open the data folder", error);
              dialog.showErrorBox(
                t("dialogs.errorTitle"),
                t("dialogs.openDataFolderFailed", { error: error.message })
              );
            }
          },
        },
        { type: "separator" },
        {
          label: t("menu.help.clearAllData"),
          click: async () => {
            try {
              if (!mainWindow) {
                logWarn("Cannot clear data: no main window");
                return;
              }

              const dataDir = getDataDirectory();

              // Show confirmation dialog
              const result = await dialog.showMessageBox(mainWindow, {
                type: "warning",
                buttons: [t("dialogs.cancel"), t("dialogs.clearAll.confirm")],
                defaultId: 0,
                cancelId: 0,
                title: t("dialogs.clearAll.title"),
                message: t("dialogs.clearAll.message"),
                detail: t("dialogs.clearAll.detail"),
              });

              if (result.response === 1) {
                // User clicked "Delete All Data"
                log("User confirmed data deletion. Clearing all data...");

                // Stop the backend first — graceful so Postgres runs
                // stop_postgres and releases the data-dir locks we are about to
                // delete; killBackend is the hard tree-kill fallback (#216).
                if (backendProcess) {
                  log("Stopping backend process...");
                  requestStop(backendProcess);
                  await gracefulShutdown(backendProcess, { killFn: killBackend });
                  backendProcess = null;
                }

                // Wait a bit for the OS to release any lingering file handles
                await new Promise((resolve) => setTimeout(resolve, 1000));

                // Delete the data directory
                if (fs.existsSync(dataDir)) {
                  try {
                    fs.rmSync(dataDir, { recursive: true, force: true });
                    log(`Successfully deleted data directory: ${dataDir}`);
                  } catch (error) {
                    logError(`Failed to delete the data directory ${dataDir}`, error);
                    throw error;
                  }
                }

                // Show success message
                await dialog.showMessageBox(mainWindow, {
                  type: "info",
                  buttons: [t("dialogs.ok")],
                  title: t("dialogs.dataCleared.title"),
                  message: t("dialogs.dataCleared.message"),
                  detail: t("dialogs.dataCleared.detail"),
                });

                // Quit the app
                app.quit();
              } else {
                log("User cancelled data deletion");
              }
            } catch (error) {
              logError("Failed to clear the data", error);
              dialog.showErrorBox(
                t("dialogs.errorTitle"),
                t("dialogs.clearAll.failedWithError", { error: error.message })
              );
            }
          },
        },
        { type: "separator" },
        {
          label: t("menu.help.learnMore"),
          click: async () => {
            await shell
              .openExternal("https://github.com/erudi-app/erudi")
              .catch((error) => logWarn("Could not open the project page in the browser", error));
          },
        },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
};

const createWindow = () => {
  if (isCreatingWindow || mainWindow) {
    log("Window creation already in progress or window exists, skipping...");
    return;
  }

  if (!app.isReady()) {
    log("createWindow called before app ready; deferring until ready event.");
    return;
  }

  isCreatingWindow = true;
  log("Creating main window...");

  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    title: "erudi - BETA",
    webPreferences: {
      preload: MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY,
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      webSecurity: true,
    },
    autoHideMenuBar: true,
    // Personnalisation de la fenêtre - boutons à droite style Windows
    titleBarStyle: "default",
    frame: true,
    icon:
      process.platform !== "darwin"
        ? path.join(__dirname, "..", "assets", "icons", "icon.png")
        : undefined,
  });

  // External links (window.open / target="_blank") must reach the system
  // browser instead of spawning a bare second Electron window.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://") || url.startsWith("http://")) {
      shell.openExternal(url);
    }
    return { action: "deny" };
  });

  mainWindow.on("closed", () => {
    log("Main window closed");
    mainWindow = null;
    isCreatingWindow = false;
  });

  // A hung renderer is the app not working, from the user's chair; both
  // transitions are recorded so the log says how long it lasted.
  mainWindow.webContents.on("unresponsive", () => {
    logError("The window's renderer is not responding");
  });
  mainWindow.webContents.on("responsive", () => {
    log("The window's renderer is responding again");
  });

  mainWindow.webContents.on("dom-ready", () => {
    mainWindow.webContents
      .executeJavaScript(
        `
      // Block navigation everywhere, but let events bubble to React
      ['dragover','drop'].forEach(type =>
        window.addEventListener(type, e => e.preventDefault(), false)
      );
    `
      )
      .catch((error) => logWarn("Could not install the drop-navigation guard", error));
  });

  mainWindow.webContents.on("will-navigate", (event) => {
    event.preventDefault();
  });

  mainWindow.webContents.session
    .clearCache()
    .catch((error) => logWarn("Could not clear the session cache", error));

  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    // Skip header modification for backend API responses to preserve
    // chunked transfer-encoding and avoid buffering streaming responses.
    if (details.url.includes("/erudi/")) {
      callback({ cancel: false });
      return;
    }

    // `img-src` deliberately omits `https:`. The window has no use for remote
    // images (the logo is bundled, attachments are data: URLs), and allowing the
    // scheme would let a markdown image link written by a model, or carried by a
    // knowledge-base document, fetch a remote host and reveal the user's IP to
    // it. Keep the scheme out; webpack.config.js pins the same policy for the
    // packaged file:// document.
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        "Content-Security-Policy": [
          "default-src 'self'; connect-src 'self' http://127.0.0.1:* http://localhost:*; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:;",
        ],
      },
    });
  });

  if (app.isPackaged) {
    mainWindow.loadFile(MAIN_WINDOW_RENDERER_INDEX);
  } else {
    mainWindow.loadURL(RENDERER_DEV_URL);
  }

  if (process.env.NODE_ENV === "development") {
    mainWindow.webContents.openDevTools();
  }

  isCreatingWindow = false;
};

// Chromium sandbox stays ON everywhere it can (#89): the switch below was a
// blanket workaround for the classic Electron sandbox launch failure on Linux
// (unprivileged user namespaces restricted on several distros, notably Ubuntu
// 24.04's AppArmor policy, which breaks AppImage startup). Keep it scoped to
// Linux only; macOS and Windows run fully sandboxed renderers.
if (process.platform === "linux") {
  app.commandLine.appendSwitch("no-sandbox");
}

// Backend readiness + diagnostics for the renderer. getInfo lets the renderer
// recover the resolved port / ready state if it mounted after the events fired
// (race-safe); restart actually re-spawns the backend (used by the Retry button).
ipcMain.handle("backend:getInfo", () => ({ port: resolvedPort, ready: backendIsReady }));
ipcMain.handle("app:getLogPath", () => logFile);

// ── Diagnostics ───────────────────────────────────────────────────────────────
// What the Diagnostics page needs from the main process. All of it is read
// locally and handed to the window; nothing is sent anywhere.

// The app's own identity. The backend has no version of its own, so the one
// electron-builder stamped into package.json is the app version, and it is
// still available when the backend never started -- which is the report we
// most want to receive.
ipcMain.handle("app:getInfo", () => ({
  version: app.getVersion(),
  platform: process.platform,
  arch: process.arch,
  electron: process.versions.electron,
  packaged: app.isPackaged,
  appLogPath: logFile,
}));

// Last WARNING/ERROR records of the app log, parsed by the same rules the
// backend applies to its own file (src/utils/appLogTail.js).
ipcMain.handle("diagnostics:appLogTail", (_event, limit) => {
  try {
    const count = Number.isInteger(limit) && limit > 0 && limit <= 500 ? limit : undefined;
    return readAppLogTail(logFile, count);
  } catch (error) {
    logWarn("Could not read the app log tail for the Diagnostics panel", error);
    return [];
  }
});

/**
 * Directories a log file is allowed to live in.
 *
 * `shell.showItemInFolder` takes whatever path it is given, and the path the
 * renderer passes comes from an HTTP response. Rather than trust it, the
 * handler checks the file sits in a directory this app writes logs to. The
 * list mirrors `backend/src/launcher/runtime_paths.py`.
 */
function knownLogDirectories() {
  const dirs = [path.dirname(logFile)];
  const appName = "erudi";
  if (process.platform === "darwin") {
    dirs.push(path.join(os.homedir(), "Library", "Logs", appName));
  } else if (process.platform === "win32") {
    const localAppData = process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local");
    dirs.push(path.join(localAppData, appName, "logs"));
  } else {
    const stateHome = process.env.XDG_STATE_HOME || path.join(os.homedir(), ".local", "state");
    dirs.push(path.join(stateHome, appName, "logs"));
  }
  if (process.env.ERUDI_DATA_ROOT) {
    dirs.push(path.join(process.env.ERUDI_DATA_ROOT, "logs"));
  }
  if (!app.isPackaged) {
    // Dev: backend/logs sits next to the frontend directory Electron runs from.
    dirs.push(path.resolve(app.getAppPath(), "..", "backend", "logs"));
  }
  return dirs.map((dir) => path.resolve(dir));
}

// Reveal a log file in the OS file manager. Called with the backend log path
// the backend reported, or with nothing, in which case the app log is used.
// A path outside a known log directory is refused and the app log is revealed
// instead -- never an arbitrary path.
ipcMain.handle("logs:reveal", (_event, filePath) => {
  let target = logFile;
  if (typeof filePath === "string" && filePath) {
    const resolved = path.resolve(filePath);
    if (knownLogDirectories().includes(path.dirname(resolved))) {
      target = resolved;
    } else {
      logWarn(`Refused to reveal a path outside the log directories: ${resolved}`);
    }
  }
  try {
    shell.showItemInFolder(target);
    return { success: true, path: target };
  } catch (error) {
    logError(`Failed to reveal the log file ${target}`, error);
    return { success: false, error: error.message };
  }
});

// Renderer log bridge: the renderer forwards its logger calls here (fire-and-
// forget ipcRenderer.send, see preload.js logAPI) so they persist in the same
// file QA already reads. Entries are validated defensively — a malformed
// payload is dropped, never thrown on.
const RENDERER_LOG_MAX_CHARS = 4000;
const asLogString = (value, max) => (typeof value === "string" ? value.slice(0, max) : "");
ipcMain.on("renderer-log", (_event, entry) => {
  try {
    if (!entry || typeof entry !== "object") return;
    const ns = asLogString(entry.ns, 120) || "unknown";
    const level = (asLogString(entry.level, 10) || "info").toUpperCase();
    const msg = asLogString(entry.msg, RENDERER_LOG_MAX_CHARS);
    const data = asLogString(entry.data, RENDERER_LOG_MAX_CHARS);
    log(`[renderer:${ns}] ${level} ${msg}${data ? ` ${data}` : ""}`);
  } catch (_) {
    // Logging must never crash the main process.
  }
});
ipcMain.handle("backend:restart", async () => {
  log("Renderer requested a backend restart.");
  // Graceful first (stop_postgres releases the data-dir locks) so the respawn
  // isn't racing an orphaned postmaster; killBackend is the hard fallback (#216).
  requestStop(backendProcess);
  await gracefulShutdown(backendProcess, { killFn: killBackend });
  backendProcess = null;
  backendIsReady = false;
  resolvedPort = null;
  startBackendSupervised();
  return { ok: true };
});

// Renderer -> main: the interface language changed (or was resolved at boot).
ipcMain.on("language:set", (_event, code) => {
  if (i18n.setLanguage(code)) {
    log(`Interface language set to ${i18n.language}; rebuilding the application menu`);
    createApplicationMenu();
  }
});

ipcMain.handle("dialog:openDirectory", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory"],
  });
  return result.filePaths[0];
});

ipcMain.handle("fs:readImageAsDataURL", async (_event, filePath) => {
  try {
    const data = fs.readFileSync(filePath);
    const ext = path.extname(filePath).slice(1).toLowerCase();
    const mime = ext === "jpg" ? "jpeg" : ext || "png";
    return `data:image/${mime};base64,${data.toString("base64")}`;
  } catch (error) {
    // The conversation renders without this attachment: degraded, and the
    // path says which one (a file the user moved or deleted, usually).
    logWarn(`Could not read the image attachment ${filePath}`, error);
    return null;
  }
});

// Persist a pasted (clipboard) image to disk and return its absolute path.
// The renderer has no fs access, and a clipboard image has no source path, so
// it would otherwise be stored as a bare [image] placeholder and lost on reload
// (#136). Writing it under the user-data dir gives it a real path that flows
// through the same [image_path:...] persistence + fs:readImageAsDataURL reload
// pipeline as any file attachment.
ipcMain.handle("image:savePasted", async (_event, dataUrl) => {
  try {
    const match = /^data:image\/([a-zA-Z0-9.+-]+);base64,(.*)$/.exec(dataUrl || "");
    if (!match) {
      return null;
    }
    let ext = match[1].toLowerCase();
    if (ext === "jpeg") {
      ext = "jpg";
    } else if (ext === "svg+xml") {
      ext = "svg";
    }
    const bytes = Buffer.from(match[2], "base64");
    const dir = path.join(getDataDirectory(), "pasted-images");
    fs.mkdirSync(dir, { recursive: true });
    const filePath = path.join(
      dir,
      `paste-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.${ext}`
    );
    fs.writeFileSync(filePath, bytes);
    return filePath;
  } catch (error) {
    // The pasted image is then kept as a bare placeholder and lost on reload.
    logError("Failed to save the pasted image to disk", error);
    return null;
  }
});

// Helper function to get the user data directory path (cross-platform)
function getDataDirectory() {
  const appName = "erudi";
  if (process.platform === "win32") {
    const localAppData = process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local");
    return path.join(localAppData, appName);
  }
  return path.join(os.homedir(), "Library", "Application Support", appName);
}

// IPC handler to open data folder in Finder
ipcMain.handle("data:openFolder", async () => {
  try {
    const dataDir = getDataDirectory();

    // Create directory if it doesn't exist
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }

    // Open in Finder
    shell.openPath(dataDir);
    log(`Opened data folder: ${dataDir}`);
    return { success: true, path: dataDir };
  } catch (error) {
    logError("Failed to open the data folder", error);
    return { success: false, error: error.message };
  }
});

// IPC handler to clear all data and quit
ipcMain.handle("data:clearAll", async () => {
  try {
    const dataDir = getDataDirectory();

    // Show confirmation dialog
    const result = await dialog.showMessageBox(mainWindow, {
      type: "warning",
      buttons: [t("dialogs.cancel"), t("dialogs.clearAll.confirm")],
      defaultId: 0,
      cancelId: 0,
      title: t("dialogs.clearAll.title"),
      message: t("dialogs.clearAll.message"),
      detail: t("dialogs.clearAll.detail"),
    });

    if (result.response === 1) {
      // User clicked "Delete All Data"
      log("User confirmed data deletion. Clearing all data...");

      // Stop the backend BEFORE deleting the data dir it holds open. Graceful
      // first so the embedded Postgres runs stop_postgres and releases its
      // locks; killBackend is the hard fallback that tears down the whole tree
      // (not a bare SIGTERM, which on Windows only hits the parent — #147/#216).
      if (backendProcess) {
        log("Stopping backend process...");
        requestStop(backendProcess);
        await gracefulShutdown(backendProcess, { killFn: killBackend });
        backendProcess = null;
      }

      // Wait a bit for the OS to release any lingering file handles
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Delete the data directory
      if (fs.existsSync(dataDir)) {
        try {
          fs.rmSync(dataDir, { recursive: true, force: true });
          log(`Successfully deleted data directory: ${dataDir}`);
        } catch (error) {
          logError(`Failed to delete the data directory ${dataDir}`, error);
          throw error;
        }
      }

      // Show success message
      await dialog.showMessageBox(mainWindow, {
        type: "info",
        buttons: [t("dialogs.ok")],
        title: t("dialogs.dataCleared.title"),
        message: t("dialogs.dataCleared.message"),
        detail: t("dialogs.dataCleared.detail"),
      });

      // Quit the app
      app.quit();

      return { success: true };
    } else {
      log("User cancelled data deletion");
      return { success: false, cancelled: true };
    }
  } catch (error) {
    logError("Failed to clear the data", error);

    await dialog.showMessageBox(mainWindow, {
      type: "error",
      buttons: [t("dialogs.ok")],
      title: t("dialogs.errorTitle"),
      message: t("dialogs.clearAll.failed"),
      detail: error.message,
    });

    return { success: false, error: error.message };
  }
});

// ── Auto-updater IPC ──────────────────────────────────────────────────────────
// Renderer can trigger an immediate install via "updater:install-now".
ipcMain.handle("updater:install-now", () => {
  if (autoUpdater) {
    autoUpdater.quitAndInstall(false, true);
  }
});

function setupAutoUpdater() {
  if (!autoUpdater) {
    return;
  }

  const send = (event, payload) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("updater-event", { event, ...payload });
    }
  };

  autoUpdater.on("checking-for-update", () => {
    log("Updater: checking for update...");
  });

  autoUpdater.on("update-available", (info) => {
    log(`Updater: update available - v${info.version}`);
    send("update-available", { version: info.version, releaseNotes: info.releaseNotes || "" });
  });

  autoUpdater.on("update-not-available", () => {
    log("Updater: already on latest version.");
  });

  autoUpdater.on("download-progress", (progress) => {
    log(`Updater: downloading... ${Math.round(progress.percent)}%`);
    send("download-progress", { percent: Math.round(progress.percent) });
  });

  autoUpdater.on("update-downloaded", (info) => {
    log(`Updater: v${info.version} downloaded, ready to install.`);
    send("update-downloaded", { version: info.version });
  });

  autoUpdater.on("error", (err) => {
    // Never crash the app over an update failure: the feature is degraded,
    // the app is not.
    logWarn("Updater error (non-fatal)", err);
  });

  // No check yet: applyAutoUpdatePreference() starts the first one once the
  // renderer has reported whether the user allows updates at all.
}

// Renderer -> main: the persisted "automatic updates" preference, sent at boot
// and on every change. Off means off end to end: no check, no download, no
// install on quit.
ipcMain.on("updater:set-enabled", (_event, enabled) => {
  autoUpdateEnabled = enabled !== false;
  applyAutoUpdatePreference();
});

function applyAutoUpdatePreference() {
  if (!autoUpdater || autoUpdateEnabled === null) {
    return;
  }

  autoUpdater.autoDownload = autoUpdateEnabled;
  autoUpdater.autoInstallOnAppQuit = autoUpdateEnabled;

  if (!autoUpdateEnabled) {
    if (updatePollTimer) {
      clearInterval(updatePollTimer);
      updatePollTimer = null;
    }
    log("Updater: automatic updates are turned off; no check will run");
    return;
  }

  if (updatePollTimer) {
    return; // already checking on its cadence
  }

  log("Updater: automatic updates are on; checking now, then every 4 hours");
  autoUpdater.checkForUpdates().catch((err) => {
    logWarn("Updater: initial check failed", err);
  });

  updatePollTimer = setInterval(
    () => {
      autoUpdater.checkForUpdates().catch((err) => {
        logWarn("Updater: periodic check failed", err);
      });
    },
    4 * 60 * 60 * 1000
  );
}

app
  .whenReady()
  .then(async () => {
    log("App ready.");

    // Create application menu and window immediately — never block on backend startup.
    // The renderer handles the loading/error state via backend-event IPC messages.
    createApplicationMenu();
    createWindow();

    // Auto-updater: wire up events and kick off initial check (production only).
    setupAutoUpdater();

    if (!app.isPackaged) {
      // Dev mode: backend is expected to be already running via dev-start.sh.
      startRealBackend()
        .then(({ port }) => {
          resolvedPort = port;
          backendIsReady = true;
          if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send("backend-event", { event: "backend_ready", port });
          }
        })
        .catch((err) => log(`Dev backend not available: ${err.message}`));
      return;
    }

    // Production: supervise the backend in the background so the window stays
    // immediately usable (the renderer shows the loading/error state via events).
    startBackendSupervised();
  })
  .catch((error) => logError("Application start-up failed", error));

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0 && !mainWindow) {
    createWindow();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    // On non-macOS, closing all windows means quit. Let before-quit own the
    // backend teardown (graceful, then hard fallback) — just ask to quit.
    app.quit();
  }
  // On macOS: app stays alive in the dock after window close (standard convention).
  // Keep the backend running so re-clicking the dock icon reconnects instantly.
});

app.on("before-quit", (e) => {
  // Graceful backend shutdown before we actually quit: close stdin and give the
  // backend up to 8s to run its lifespan (checkpointer close, stop_postgres),
  // else killBackend hard-kills the tree (#216). Deferring the quit once (via
  // preventDefault) is the only way to await this; the re-issued app.quit()
  // finds shuttingDown=true and this handler falls through.
  if (!shuttingDown && backendProcess) {
    e.preventDefault();
    shuttingDown = true;
    log("Stopping backend process before quit...");
    requestStop(backendProcess);
    gracefulShutdown(backendProcess, { killFn: killBackend }).finally(() => {
      backendProcess = null;
      app.quit();
    });
  }
});
