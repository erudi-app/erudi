import { describe, it, expect, vi, afterEach } from "vitest";
import i18n from "../i18n";
import { exportConversation, exportLabels } from "./conversationExport";

// The renderer side of the Markdown export (#430): fetch the conversation,
// build the text, hand title + text to the main process, which owns the dialog
// and the file. No path ever leaves the renderer.

const conversation = {
  id: 7,
  name: "Airships",
  created_at: "2026-11-03T10:00:00",
  messages: [
    { id: 1, sender: "user", content: "Hello" },
    { id: 2, sender: "llm", content: "Hi", trace: [{ t: "thinking", text: "HIDDEN" }] },
  ],
};

const tr = (key, options = {}) => (options.count !== undefined ? `${key}#${options.count}` : key);

afterEach(() => {
  delete globalThis.window;
});

function setup({ ok = true, save = vi.fn(async () => ({ status: "saved" })) } = {}) {
  globalThis.window = { exportAPI: { saveMarkdown: save } };
  const fetcher = vi.fn(async () => ({
    ok,
    status: ok ? 200 : 500,
    json: async () => conversation,
  }));
  return { save, fetcher };
}

describe("exportConversation", () => {
  it("fetches the conversation and sends only its title and Markdown text to main", async () => {
    const { save, fetcher } = setup();
    const result = await exportConversation(7, { t: tr, fetcher });

    expect(String(fetcher.mock.calls[0][0])).toMatch(/\/conversations\/7$/);
    expect(save).toHaveBeenCalledTimes(1);
    const payload = save.mock.calls[0][0];
    expect(Object.keys(payload).sort()).toEqual(["text", "title"]);
    expect(payload.title).toBe("Airships");
    expect(payload.text).toContain("# Airships");
    expect(payload.text).toContain("Hello");
    expect(payload.text).not.toContain("HIDDEN");
    expect(result).toEqual({ status: "saved" });
  });

  it("passes a cancelled dialog back as is", async () => {
    const { fetcher } = setup({ save: vi.fn(async () => ({ status: "cancelled" })) });
    expect(await exportConversation(7, { t: tr, fetcher })).toEqual({ status: "cancelled" });
  });

  it("throws when the conversation cannot be fetched, and never opens the dialog", async () => {
    const { save, fetcher } = setup({ ok: false });
    // The interface's sentence for the status, never English text (#780).
    await expect(exportConversation(7, { t: tr, fetcher })).rejects.toThrow("API Error: 500");
    expect(save).not.toHaveBeenCalled();
  });

  it("throws when main reports a failure", async () => {
    const { fetcher } = setup({ save: vi.fn(async () => ({ status: "error" })) });
    // A sentence of the interface, never English text (#780).
    await expect(exportConversation(7, { t: tr, fetcher })).rejects.toThrow(
      "chat:export.errors.saveFailed"
    );
  });

  it("throws when the bridge is missing (a plain browser)", async () => {
    globalThis.window = {};
    const fetcher = vi.fn(async () => ({ ok: true, json: async () => conversation }));
    await expect(exportConversation(7, { t: tr, fetcher })).rejects.toThrow(
      "chat:export.errors.bridgeUnavailable"
    );
  });

  it("never exports an ephemeral conversation (#428)", async () => {
    const save = vi.fn(async () => ({ status: "saved" }));
    globalThis.window = { exportAPI: { saveMarkdown: save } };
    const fetcher = vi.fn(async () => ({
      ok: true,
      json: async () => ({ ...conversation, ephemeral: true }),
    }));

    await expect(exportConversation(7, { t: tr, fetcher })).rejects.toThrow(
      "chat:export.errors.ephemeral"
    );
    expect(save).not.toHaveBeenCalled();
  });
});

describe("exportLabels", () => {
  afterEach(async () => {
    await i18n.changeLanguage("en");
  });

  it("writes an answer's notes in the interface language (#764)", async () => {
    await i18n.changeLanguage("fr");
    const english = "This model doesn't support images — your image was ignored.";
    const trace = [{ t: "notice", kind: "images_ignored", params: {} }];

    const answer = exportLabels(conversation, tr).answer(`*${english}*\n\nUn carré.`, trace);

    expect(answer).toBe(
      "*Ce modèle ne prend pas en charge les images : votre image a été ignorée.*\n\nUn carré."
    );
  });
});
