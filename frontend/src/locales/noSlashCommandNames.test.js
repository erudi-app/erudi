/**
 * Slash command names are a typing shortcut of the message box, not interface
 * vocabulary (#704): no catalog string of any namespace names one. The
 * composer's drop-up draws its names from the command registry
 * (`utils/slashCommands.js`), not from a catalog, and
 * `chat:commands.needsConversation` answers a command the user typed by
 * interpolating it (`{{command}}`), so nothing needs an allowlist: every leaf
 * of every namespace of every language is scanned. A closing tag such as
 * `</model>` is markup, not a command (the character before the slash is `<`).
 */
import { describe, it, expect } from "vitest";
import { NAMESPACES, resources } from "../i18n/resources";
import { SUPPORTED_LANGUAGES } from "../i18n/languages";

const SLASH_NAME = /(^|[^<\w])\/(config|model|effort|compact|context)\b/;

function leaves(obj, prefix = "", out = {}) {
  for (const [key, value] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value !== null && typeof value === "object") leaves(value, path, out);
    else out[path] = String(value);
  }
  return out;
}

describe("slash command names in the interface copy", () => {
  for (const lang of SUPPORTED_LANGUAGES) {
    for (const ns of NAMESPACES) {
      it(`${lang}/${ns} names none`, () => {
        const offending = Object.entries(leaves(resources[lang][ns]))
          .filter(([, text]) => SLASH_NAME.test(text))
          .map(([key]) => key);
        expect(offending).toEqual([]);
      });
    }

    it(`${lang}: the typed-command answer interpolates the command`, () => {
      expect(resources[lang].chat.commands.needsConversation).toContain("{{command}}");
    });
  }
});
