import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createMainTranslator } from "../i18n/mainTranslator.js";
import { buildMenuTemplate } from "./applicationMenu.js";

const LANGUAGES = ["en", "fr", "es", "zh"];
const handlers = {
  openDataFolder: () => {},
  clearAllData: () => {},
  clearAllDataVisible: true,
  learnMore: () => {},
};

function build(language, isMac) {
  const i18n = createMainTranslator(language);
  return buildMenuTemplate({
    t: (key, vars) => i18n.t(key, vars),
    isMac,
    appName: "Erudi",
    handlers,
  });
}

// Every entry of the tree, submenus included.
function walk(items, out = []) {
  for (const item of items) {
    out.push(item);
    if (Array.isArray(item.submenu)) walk(item.submenu, out);
  }
  return out;
}

const entriesWithRole = (template) => walk(template).filter((item) => item.role);

describe("buildMenuTemplate", () => {
  for (const language of LANGUAGES) {
    for (const isMac of [true, false]) {
      describe(`${language} (${isMac ? "mac" : "win/linux"})`, () => {
        const template = build(language, isMac);

        it("gives every role entry a translated label", () => {
          const roled = entriesWithRole(template);
          expect(roled.length).toBeGreaterThan(10);
          for (const item of roled) {
            expect(typeof item.label, `role ${item.role}`).toBe("string");
            expect(item.label, `role ${item.role}`).not.toBe("");
            expect(item.label, `role ${item.role}`).not.toMatch(/^menu\./);
          }
        });

        it("never writes the app name in lowercase", () => {
          for (const item of walk(template)) {
            expect(item.label ?? "").not.toMatch(/erudi/);
          }
        });

        it("titles the Help menu", () => {
          const help = template.find((item) => item.role === "help");
          expect(help.label).toBeTruthy();
        });
      });
    }
  }

  it("reads Quit Erudi in English and Quitter Erudi in French", () => {
    const quit = (template) => entriesWithRole(template).find((i) => i.role === "quit").label;
    expect(quit(build("en", true))).toBe("Quit Erudi");
    expect(quit(build("fr", true))).toBe("Quitter Erudi");
  });

  it("writes Exit, without the app name, in the non-mac File menu", () => {
    const exit = (language) => {
      const file = build(language, false).find((item) => item.submenu?.[0]?.role === "quit");
      return file.submenu[0].label;
    };
    expect(exit("en")).toBe("Exit");
    expect(exit("fr")).toBe("Quitter");
    expect(exit("es")).toBe("Salir");
    expect(exit("zh")).toBe("退出");
    expect(entriesWithRole(build("en", false)).some((i) => /Quit/.test(i.label))).toBe(false);
  });

  it("writes Close Window on mac and a plain Close elsewhere", () => {
    const close = (isMac) => entriesWithRole(build("en", isMac)).find((i) => i.role === "close");
    expect(close(true).label).toBe("Close Window");
    expect(close(false).label).toBe("Close");
  });

  it("reads the French app menu with the app name", () => {
    const labels = Object.fromEntries(
      entriesWithRole(build("fr", true)).map((i) => [i.role, i.label])
    );
    expect(labels.about).toBe("À propos d'Erudi");
    expect(labels.hide).toBe("Masquer Erudi");
    expect(labels.hideOthers).toBe("Masquer les autres");
    expect(labels.unhide).toBe("Tout afficher");
  });

  it("names the application menu Erudi, not the package name", () => {
    expect(build("fr", true)[0].label).toBe("Erudi");
  });

  it("keeps every role of the native menu", () => {
    const roles = (isMac) => entriesWithRole(build("en", isMac)).map((i) => i.role);
    expect(roles(true)).toEqual([
      "about",
      "services",
      "hide",
      "hideOthers",
      "unhide",
      "quit",
      "close",
      "undo",
      "redo",
      "cut",
      "copy",
      "paste",
      "pasteAndMatchStyle",
      "delete",
      "selectAll",
      "startSpeaking",
      "stopSpeaking",
      "reload",
      "forceReload",
      "toggleDevTools",
      "resetZoom",
      "zoomIn",
      "zoomOut",
      "togglefullscreen",
      "minimize",
      "zoom",
      "front",
      "window",
      "help",
    ]);
    expect(roles(false)).toEqual([
      "quit",
      "undo",
      "redo",
      "cut",
      "copy",
      "paste",
      "delete",
      "selectAll",
      "reload",
      "forceReload",
      "toggleDevTools",
      "resetZoom",
      "zoomIn",
      "zoomOut",
      "togglefullscreen",
      "minimize",
      "zoom",
      "close",
      "help",
    ]);
  });

  it("wires each Help entry to its handler", () => {
    const help = build("en", false).find((item) => item.role === "help");
    const entry = (label) => help.submenu.find((i) => i.label === label);
    expect(entry("Open Data Folder").click).toBe(handlers.openDataFolder);
    expect(entry("Clear All Data...").click).toBe(handlers.clearAllData);
    expect(entry("Clear All Data...").visible).toBe(true);
    expect(entry("Learn More").click).toBe(handlers.learnMore);
  });

  it("hides Clear All Data when the handlers say so", () => {
    const i18n = createMainTranslator("en");
    const template = buildMenuTemplate({
      t: i18n.t,
      isMac: false,
      appName: "Erudi",
      handlers: { ...handlers, clearAllDataVisible: false },
    });
    const help = template.find((item) => item.role === "help");
    expect(help.submenu.find((i) => i.label === "Clear All Data...").visible).toBe(false);
  });
});

describe("main.js menu wiring", () => {
  const source = readFileSync(fileURLToPath(new URL("../main.js", import.meta.url)), "utf8");

  it("passes the display name, never app.name, to the menu", () => {
    expect(source).toMatch(/const APP_DISPLAY_NAME = "Erudi";/);
    expect(source).toMatch(/buildMenuTemplate\(\{[^}]*appName: APP_DISPLAY_NAME,/s);
    expect(source).not.toMatch(/appName: app\.name/);
  });
});
