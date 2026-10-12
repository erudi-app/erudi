// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// #607 — the Settings page's "Default compaction threshold" card: sets the
// threshold new conversations inherit (`default_compaction_threshold` on
// `GET/PUT /user_settings/`); each conversation then owns its own control
// (see ConversationPage / HeaderBar), so changing this never retro-affects an
// existing conversation.

const { getMock, putMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  putMock: vi.fn(),
}));

vi.mock("../services/api/client", () => ({
  default: { get: getMock, put: putMock },
  apiClient: { get: getMock, put: putMock },
}));

vi.mock("../components/Sidebar", () => ({
  default: () => <div data-testid="sidebar" />,
}));

import SettingsPage from "./SettingsPage";
import { resources } from "../i18n/resources";
import { SUPPORTED_LANGUAGES } from "../i18n/languages";

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/erudi/settings"]}>
      <SettingsPage />
    </MemoryRouter>
  );
}

function mockGetByUrl({ userSettings, appStartup }) {
  getMock.mockImplementation((url) => {
    if (url === "/hardware/app_startup") {
      return Promise.resolve(appStartup);
    }
    return Promise.resolve(userSettings);
  });
}

const SETTINGS = {
  web_search_enabled: false,
  language: "en",
  auto_update_enabled: true,
  inference_backend: "auto",
  default_reasoning_effort: "medium",
  default_compaction_threshold: 80,
};

beforeEach(() => {
  mockGetByUrl({ userSettings: SETTINGS, appStartup: { backend_type: "cuda" } });
  putMock.mockResolvedValue({ ...SETTINGS, default_compaction_threshold: 70 });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("SettingsPage — Default compaction threshold (#607)", () => {
  it("renders the card with 50 % to 90 % in steps of 5 and the current value", async () => {
    renderPage();
    expect(await screen.findByText("Default compaction threshold")).toBeTruthy();
    const select = screen.getByLabelText("Default compaction threshold");
    const values = Array.from(select.querySelectorAll("option")).map((o) => o.value);
    expect(values).toEqual(["50", "55", "60", "65", "70", "75", "80", "85", "90"]);
    await waitFor(() => expect(select.value).toBe("80"));
  });

  it("defaults to 80 % when the field is absent", async () => {
    const withoutField = { ...SETTINGS };
    delete withoutField.default_compaction_threshold;
    mockGetByUrl({ userSettings: withoutField, appStartup: { backend_type: "cuda" } });
    renderPage();
    const select = await screen.findByLabelText("Default compaction threshold");
    await waitFor(() => expect(select.value).toBe("80"));
  });

  it("PUTs the new value, as a number, when changed and updates the UI", async () => {
    renderPage();
    const select = await screen.findByLabelText("Default compaction threshold");
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    fireEvent.change(select, { target: { value: "70" } });

    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith("/user_settings/", { default_compaction_threshold: 70 })
    );
    await waitFor(() => expect(select.value).toBe("70"));
  });

  it("explains what the threshold does and that only new conversations inherit it", async () => {
    renderPage();
    await screen.findByText("Default compaction threshold");
    const page = document.body.textContent;
    expect(page).toMatch(/summarized/i);
    expect(page).toMatch(/never affects a conversation that already exists/i);
  });
});

describe("compaction threshold copy (#607)", () => {
  it("exists in every language", () => {
    for (const lang of SUPPORTED_LANGUAGES) {
      const { chat, settings } = resources[lang];
      expect(chat.header.compactionThreshold, lang).toBeTruthy();
      expect(chat.header.tooltips.compactionThreshold, lang).toBeTruthy();
      expect(settings.compactionThreshold.title, lang).toBeTruthy();
      expect(settings.compactionThreshold.description, lang).toBeTruthy();
      expect(settings.compactionThreshold.note, lang).toBeTruthy();
      expect(settings.compactionThreshold.selectLabel, lang).toBeTruthy();
    }
  });
});
