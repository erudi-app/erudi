/* eslint-disable no-console */
// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts

const { contextBridge, ipcRenderer, webUtils } = require("electron");

// Vérifie si preload.js est bien chargé
console.log("Le script preload.js est chargé");

contextBridge.exposeInMainWorld("electron", {
  // Vérifie si l'API est exposée
  openDirectory: () => {
    return ipcRenderer.invoke("dialog:openDirectory");
  },
  getFilePath: (file) => {
    if (webUtils?.getPathForFile) {
      const path = webUtils.getPathForFile(file);
      console.log("webUtils.getPathForFile returned:", path);
      return path;
    }

    console.log("Falling back to file.path:", file.path);
    return file.path;
  },
});

// Local filesystem image loader (for reloading image attachments from stored paths)
contextBridge.exposeInMainWorld("fsAPI", {
  readImageAsDataURL: (filePath) => ipcRenderer.invoke("fs:readImageAsDataURL", filePath),
});

// Pasted-image persistence: writes a clipboard image's bytes to disk and returns
// its absolute path, so a pasted attachment survives reload like a file one (#136).
contextBridge.exposeInMainWorld("imageAPI", {
  savePasted: (dataUrl) => ipcRenderer.invoke("image:savePasted", dataUrl),
});

// Expose additional API for data management
contextBridge.exposeInMainWorld("electronAPI", {
  openDataFolder: () => ipcRenderer.invoke("data:openFolder"),
  clearAllData: () => ipcRenderer.invoke("data:clearAll"),
});

// Backend lifecycle bridge: forwards run.py / main.js "backend-event" messages
// ({event: "starting"|"ready"|"shutdown"|"startup_error", code?, message?}) so the
// renderer can show a real error instead of an endless loading spinner.
contextBridge.exposeInMainWorld("backendAPI", {
  onBackendEvent: (callback) => {
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on("backend-event", handler);
    return () => ipcRenderer.removeListener("backend-event", handler);
  },
  // Recover {port, ready} if the renderer mounted after the events fired.
  getInfo: () => ipcRenderer.invoke("backend:getInfo"),
  // Actually re-spawn the backend (used by the error screen's Retry button).
  restartBackend: () => ipcRenderer.invoke("backend:restart"),
  // OS-correct path to the backend log, for the error screen.
  getLogPath: () => ipcRenderer.invoke("app:getLogPath"),
});

// Diagnostics bridge: what the Diagnostics page needs from the main process —
// the app's own version and platform, the last WARNING/ERROR records of the
// app log, and revealing a log file in the OS file manager. Read-only and
// local; the page sends nothing anywhere.
contextBridge.exposeInMainWorld("diagnosticsAPI", {
  getAppInfo: () => ipcRenderer.invoke("app:getInfo"),
  appLogTail: (limit) => ipcRenderer.invoke("diagnostics:appLogTail", limit),
  // Pass the backend log path to reveal that one; omit it for the app log.
  // Main refuses a path outside its known log directories.
  revealLog: (filePath) => ipcRenderer.invoke("logs:reveal", filePath),
});

// Renderer log bridge: fire-and-forget forwarding of renderer logger entries
// ({ts, level, ns, msg, data}) to the main process, which persists them in the
// same log file QA already reads (see main.js "renderer-log" handler).
contextBridge.exposeInMainWorld("logAPI", {
  send: (entry) => ipcRenderer.send("renderer-log", entry),
});

// Interface language bridge (#385): the renderer owns the choice (i18next +
// the user_settings API) and tells main so the native menu and dialogs are
// rebuilt in the same language. Fire-and-forget, sent at boot and on change.
contextBridge.exposeInMainWorld("languageAPI", {
  set: (code) => ipcRenderer.send("language:set", code),
});

// Auto-updater bridge
contextBridge.exposeInMainWorld("updaterAPI", {
  // Register a callback for updater events from main process.
  // event types: "update-available" | "download-progress" | "update-downloaded"
  onUpdaterEvent: (callback) => {
    ipcRenderer.on("updater-event", (_event, payload) => callback(payload));
    // Return cleanup function
    return () => ipcRenderer.removeAllListeners("updater-event");
  },
  // Trigger immediate quit-and-install
  installNow: () => ipcRenderer.invoke("updater:install-now"),
  // The renderer owns the persisted "automatic updates" preference (the
  // user_settings API); main owns electron-updater and holds every check until
  // this arrives. Fire-and-forget, sent at boot and whenever the user changes it.
  setAutoUpdateEnabled: (enabled) => ipcRenderer.send("updater:set-enabled", enabled),
});
