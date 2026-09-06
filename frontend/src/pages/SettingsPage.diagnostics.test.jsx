// @vitest-environment jsdom
/**
 * Diagnostics is not on the settings page. It has a page of its own behind the
 * bug button, so Settings must neither render the diagnostics content nor ask
 * the backend for it.
 */
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

const { getMock, putMock } = vi.hoisted(() => ({ getMock: vi.fn(), putMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: getMock, put: putMock },
  apiClient: { get: getMock, put: putMock },
}));

vi.mock("../components/Sidebar", () => ({
  default: () => <div data-testid="sidebar" />,
}));

import SettingsPage from "./SettingsPage";
import { SETTINGS_PATH } from "../utils/routes";

beforeEach(() => {
  getMock.mockResolvedValue({
    web_search_enabled: false,
    language: "en",
    auto_update_enabled: true,
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("SettingsPage — no Diagnostics", () => {
  it("renders none of the diagnostics content and never requests it", async () => {
    render(
      <MemoryRouter initialEntries={[SETTINGS_PATH]}>
        <SettingsPage />
      </MemoryRouter>
    );
    expect(await screen.findByRole("heading", { level: 1, name: "Settings" })).toBeTruthy();
    expect(screen.queryByText("Diagnostics")).toBeNull();
    expect(screen.queryByLabelText("Diagnostics to copy")).toBeNull();
    expect(screen.queryByRole("button", { name: "Open log folder" })).toBeNull();
    expect(getMock).not.toHaveBeenCalledWith("/diagnostics/");
  });
});
