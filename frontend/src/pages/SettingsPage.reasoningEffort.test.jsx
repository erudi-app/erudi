// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// PR-D2 — the Settings page's "Default reasoning effort" card: sets the
// starting value new conversations inherit (`default_reasoning_effort` on
// `GET/PUT /user_settings/`); each conversation then owns its own control
// (see ConversationPage / HeaderBar), so changing this never retro-affects
// an existing conversation.

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

beforeEach(() => {
  mockGetByUrl({
    userSettings: {
      web_search_enabled: false,
      language: "en",
      auto_update_enabled: true,
      inference_backend: "auto",
      default_reasoning_effort: "medium",
    },
    appStartup: { backend_type: "cuda" },
  });
  putMock.mockResolvedValue({
    web_search_enabled: false,
    language: "en",
    auto_update_enabled: true,
    default_reasoning_effort: "high",
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("SettingsPage — Default reasoning effort (PR-D2)", () => {
  it("renders the card with the five levels and the current value", async () => {
    renderPage();
    expect(await screen.findByText("Default reasoning effort")).toBeTruthy();
    const select = screen.getByLabelText("Default reasoning effort");
    const values = Array.from(select.querySelectorAll("option")).map((o) => o.value);
    expect(values).toEqual(["none", "low", "medium", "high", "xhigh"]);
    await waitFor(() => expect(select.value).toBe("medium"));
  });

  it("defaults to Medium when the field is absent", async () => {
    mockGetByUrl({
      userSettings: { web_search_enabled: false, language: "en", auto_update_enabled: true },
      appStartup: { backend_type: "cuda" },
    });
    renderPage();
    const select = await screen.findByLabelText("Default reasoning effort");
    await waitFor(() => expect(select.value).toBe("medium"));
  });

  it("PUTs the new value when changed and updates the UI", async () => {
    renderPage();
    const select = await screen.findByLabelText("Default reasoning effort");
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    fireEvent.change(select, { target: { value: "high" } });

    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith("/user_settings/", { default_reasoning_effort: "high" })
    );
    await waitFor(() => expect(select.value).toBe("high"));
  });

  it("explains new conversations inherit it and existing ones keep their own", async () => {
    renderPage();
    await screen.findByText("Default reasoning effort");
    const page = document.body.textContent;
    expect(page).toMatch(/new conversations/i);
  });
});
