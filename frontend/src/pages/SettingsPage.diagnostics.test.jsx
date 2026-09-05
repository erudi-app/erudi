// @vitest-environment jsdom
/**
 * The Diagnostics panel's place on the settings page, and the anchor the
 * sidebar's bug button lands on.
 */
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, waitFor } from "@testing-library/react";
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
import { DIAGNOSTICS_PATH } from "../utils/routes";

const renderAt = (entry) =>
  render(
    <MemoryRouter initialEntries={[entry]}>
      <SettingsPage />
    </MemoryRouter>
  );

beforeEach(() => {
  getMock.mockImplementation((path) => {
    if (path === "/diagnostics/") {
      return Promise.resolve({
        environment: {
          platform: "Darwin",
          engine: "MLX_Engine",
          db: "ok",
          backend_log_path: "/Users/x/Library/Logs/erudi/backend.log",
        },
        recent_errors: [],
      });
    }
    return Promise.resolve({
      web_search_enabled: false,
      language: "en",
      auto_update_enabled: true,
    });
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("SettingsPage — Diagnostics", () => {
  it("renders the Diagnostics panel under an anchorable id", async () => {
    const { container } = renderAt("/erudi/settings");
    expect(await screen.findByText("Diagnostics")).toBeTruthy();
    expect(container.querySelector("#diagnostics")).toBeTruthy();
  });

  it("scrolls to the panel when arriving on the diagnostics anchor", async () => {
    // HashRouter puts the whole route in the fragment, so the browser cannot
    // follow the second `#` itself; the page does the scroll.
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;
    renderAt(DIAGNOSTICS_PATH);
    await waitFor(() => expect(scrollIntoView).toHaveBeenCalled());
  });

  it("does not scroll when arriving on the page itself", async () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;
    renderAt("/erudi/settings");
    expect(await screen.findByText("Diagnostics")).toBeTruthy();
    expect(scrollIntoView).not.toHaveBeenCalled();
  });
});
