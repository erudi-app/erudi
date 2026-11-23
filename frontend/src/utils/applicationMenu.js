/**
 * Template of the native application menu (#763).
 *
 * Pure: no Electron import, so a test can walk it in every language. Each
 * `role` entry carries an explicit translated `label` — without one Electron
 * applies its own English string, and names the app after `app.name`
 * (`package.json` "productName", lowercase, which also decides the userData
 * folder and the single-instance lock, so it must not change). The `role` is
 * kept on every entry: it still provides the native shortcut and behaviour.
 *
 * `t(key, vars)` is the main-process translator; `appName` is the display
 * name written in the labels that carry it.
 */

/** A role entry whose label is `menu.roles.<role>`. */
const roleItem = (t, appName, role) => ({ role, label: t(`menu.roles.${role}`, { appName }) });

export function buildMenuTemplate({ t, isMac, appName, handlers }) {
  const role = (name) => roleItem(t, appName, name);

  return [
    // App menu (macOS only)
    ...(isMac
      ? [
          {
            label: appName,
            submenu: [
              role("about"),
              { type: "separator" },
              role("services"),
              { type: "separator" },
              role("hide"),
              role("hideOthers"),
              role("unhide"),
              { type: "separator" },
              role("quit"),
            ],
          },
        ]
      : []),

    // File menu
    {
      label: t("menu.file"),
      submenu: [isMac ? role("close") : role("quit")],
    },

    // Edit menu
    {
      label: t("menu.edit"),
      submenu: [
        role("undo"),
        role("redo"),
        { type: "separator" },
        role("cut"),
        role("copy"),
        role("paste"),
        ...(isMac
          ? [
              role("pasteAndMatchStyle"),
              role("delete"),
              role("selectAll"),
              { type: "separator" },
              {
                label: t("menu.speech"),
                submenu: [role("startSpeaking"), role("stopSpeaking")],
              },
            ]
          : [role("delete"), { type: "separator" }, role("selectAll")]),
      ],
    },

    // View menu
    {
      label: t("menu.view"),
      submenu: [
        role("reload"),
        role("forceReload"),
        role("toggleDevTools"),
        { type: "separator" },
        role("resetZoom"),
        role("zoomIn"),
        role("zoomOut"),
        { type: "separator" },
        role("togglefullscreen"),
      ],
    },

    // Window menu
    {
      label: t("menu.window"),
      submenu: [
        role("minimize"),
        role("zoom"),
        ...(isMac
          ? [{ type: "separator" }, role("front"), { type: "separator" }, role("window")]
          : [role("close")]),
      ],
    },

    // Help menu
    {
      role: "help",
      label: t("menu.help.title"),
      submenu: [
        {
          label: t("menu.help.openDataFolder"),
          click: handlers.openDataFolder,
        },
        { type: "separator" },
        {
          label: t("menu.help.clearAllData"),
          click: handlers.clearAllData,
          // Hidden in a run from source.
          visible: handlers.clearAllDataVisible,
        },
        { type: "separator" },
        {
          label: t("menu.help.learnMore"),
          click: handlers.learnMore,
        },
      ],
    },
  ];
}
