// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// #619 — the Settings page's "Unload the model when idle" card: how long an
// unused model stays loaded (`idle_unload_minutes` on `GET/PUT
// /user_settings/`): 5, 15, 30 or 60 minutes, or Never (stored as 0). The
// backend applies a change at once, without a relaunch.

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

const TITLE = "Unload the model when idle";

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/erudi/settings"]}>
      <SettingsPage />
    </MemoryRouter>
  );
}

function mockSettings(userSettings) {
  getMock.mockImplementation((url) =>
    Promise.resolve(url === "/hardware/app_startup" ? { backend_type: "mlx" } : userSettings)
  );
}

const SETTINGS = {
  web_search_enabled: false,
  language: "en",
  auto_update_enabled: true,
  auto_download_updates: false,
  auto_install_updates: false,
  inference_backend: "auto",
  default_reasoning_effort: "medium",
  default_compaction_threshold: 80,
  idle_unload_minutes: 5,
};

beforeEach(() => {
  mockSettings(SETTINGS);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("SettingsPage — Unload the model when idle (#619)", () => {
  it("offers 5, 15, 30 and 60 minutes and Never, with 5 minutes selected", async () => {
    renderPage();
    expect(await screen.findByText(TITLE)).toBeTruthy();
    const select = screen.getByLabelText(TITLE);
    const options = Array.from(select.querySelectorAll("option")).map((o) => [
      o.value,
      o.textContent,
    ]);
    expect(options).toEqual([
      ["5", "5 minutes"],
      ["15", "15 minutes"],
      ["30", "30 minutes"],
      ["60", "60 minutes"],
      ["0", "Never"],
    ]);
    await waitFor(() => expect(select.value).toBe("5"));
  });

  it("shows Never for a stored 0", async () => {
    mockSettings({ ...SETTINGS, idle_unload_minutes: 0 });
    renderPage();
    const select = await screen.findByLabelText(TITLE);
    await waitFor(() => expect(select.value).toBe("0"));
  });

  it("defaults to 5 minutes when the field is absent", async () => {
    const withoutField = { ...SETTINGS };
    delete withoutField.idle_unload_minutes;
    mockSettings(withoutField);
    renderPage();
    const select = await screen.findByLabelText(TITLE);
    await waitFor(() => expect(select.value).toBe("5"));
  });

  it.each([
    ["15", 15],
    ["0", 0],
  ])("PUTs %s as a number when picked and shows it", async (value, minutes) => {
    putMock.mockResolvedValue({ ...SETTINGS, idle_unload_minutes: minutes });
    renderPage();
    const select = await screen.findByLabelText(TITLE);
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    fireEvent.change(select, { target: { value } });

    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith("/user_settings/", { idle_unload_minutes: minutes })
    );
    await waitFor(() => expect(select.value).toBe(value));
  });

  it("says an unused model leaves memory after the delay and reloads at the next message", async () => {
    renderPage();
    await screen.findByText(TITLE);
    const page = document.body.textContent;
    expect(page).toMatch(/leaves memory/i);
    expect(page).toMatch(/reloads at your next message/i);
    // The Unload button is refused while an answer is being written.
    expect(page).toMatch(/whenever no answer is being written/i);
    expect(page).not.toMatch(/at any time/i);
  });
});

describe("idle unload copy (#619)", () => {
  it("exists in every language", () => {
    for (const lang of SUPPORTED_LANGUAGES) {
      const { settings } = resources[lang];
      for (const key of ["title", "description", "note", "selectLabel", "minutes", "never"]) {
        expect(settings.idleUnload[key], `${lang} ${key}`).toBeTruthy();
      }
    }
  });
});
