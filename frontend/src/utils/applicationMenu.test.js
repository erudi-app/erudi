import { describe, it, expect } from "vitest";
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
    expect(quit(build("fr", false))).toBe("Quitter Erudi");
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

  it("wires the Help actions to the handlers", () => {
    const help = build("en", false).find((item) => item.role === "help");
    const clear = help.submenu.find((i) => i.click === handlers.clearAllData);
    expect(clear.visible).toBe(true);
    expect(help.submenu.some((i) => i.click === handlers.openDataFolder)).toBe(true);
    expect(help.submenu.some((i) => i.click === handlers.learnMore)).toBe(true);
  });
});
