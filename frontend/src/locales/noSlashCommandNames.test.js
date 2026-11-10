/**
 * Slash command names are a typing shortcut of the message box, not interface
 * vocabulary (#704): the Settings page, the conversation header and the
 * conversation's settings window speak in the interface's own terms. The
 * drop-up of the composer keeps its names (they are typed there, from the
 * registry, not from a catalog), and `chat:commands.needsConversation`
 * answers a command the user typed, so it interpolates it.
 */
import { describe, it, expect } from "vitest";
import { resources } from "../i18n/resources";
import { SUPPORTED_LANGUAGES } from "../i18n/languages";

// A slash followed by a word, standing alone: `/config`, `/config global`.
// A closing tag such as `</model>` is not one (no space or start before it).
const SLASH_NAME = /(^|[\s("'])\/[a-z]+/;

function leaves(obj, prefix = "", out = {}) {
  for (const [key, value] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value !== null && typeof value === "object") leaves(value, path, out);
    else out[path] = String(value);
  }
  return out;
}

const offending = (catalog, skip = () => false) =>
  Object.entries(leaves(catalog))
    .filter(([key, text]) => !skip(key) && SLASH_NAME.test(text))
    .map(([key]) => key);

describe("slash command names in the interface copy", () => {
  for (const lang of SUPPORTED_LANGUAGES) {
    it(`${lang}: the Settings page names none`, () => {
      expect(offending(resources[lang].settings)).toEqual([]);
    });

    it(`${lang}: the header, the settings window and the model picker name none`, () => {
      const chat = resources[lang].chat;
      const scanned = {
        conversationHeader: chat.conversationHeader,
        modelPopup: chat.modelPopup,
        modelPicker: chat.modelPicker,
        config: chat.config,
        commands: Object.fromEntries(
          Object.entries(chat.commands).filter(([key]) => key !== "needsConversation")
        ),
      };
      expect(offending(scanned)).toEqual([]);
    });
  }
});
