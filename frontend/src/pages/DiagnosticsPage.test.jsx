// @vitest-environment jsdom
/**
 * The Diagnostics page: the destination of the bug button in the sidebar
 * rail. It carries the same chrome as the other pages (rail, heading) and the
 * diagnostics content; the heading wears the bug icon so the page reads as the
 * place the button leads to.
 */
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

const { getMock } = vi.hoisted(() => ({ getMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: getMock },
  apiClient: { get: getMock },
}));

vi.mock("../components/Sidebar", () => ({
  default: () => <div data-testid="sidebar" />,
}));

import DiagnosticsPage from "./DiagnosticsPage";
import { DIAGNOSTICS_PATH } from "../utils/routes";
import { getLastDiagnosticsVisit, LAST_VISIT_STORAGE_KEY } from "../utils/bugCounter";

function memoryStorage(initial = {}) {
  const store = new Map(Object.entries(initial));
  return {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
    clear: () => store.clear(),
  };
}

beforeEach(() => {
  Object.defineProperty(window, "localStorage", {
    value: memoryStorage(),
    configurable: true,
    writable: true,
  });
  getMock.mockResolvedValue({
    environment: {
      platform: "Darwin",
      engine: "MLX_Engine",
      db: "ok",
      backend_log_path: "/Users/x/Library/Logs/erudi/backend.log",
    },
    recent_errors: [],
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const renderPage = () =>
  render(
    <MemoryRouter initialEntries={[DIAGNOSTICS_PATH]}>
      <DiagnosticsPage />
    </MemoryRouter>
  );

describe("DiagnosticsPage", () => {
  it("renders the rail, a heading and the diagnostics content", async () => {
    renderPage();
    expect(screen.getByTestId("sidebar")).toBeTruthy();
    expect(screen.getByRole("heading", { level: 1, name: "Diagnostics" })).toBeTruthy();
    expect(await screen.findByText("MLX_Engine")).toBeTruthy();
    // The lighter Diagnostics rendering has no text preview, and this mock
    // has no recent errors, so there is nothing to copy either.
    expect(screen.queryByLabelText("Diagnostics to copy")).toBeNull();
    expect(screen.queryByRole("button", { name: /copy/i })).toBeNull();
    expect(screen.getByRole("button", { name: "Open log folder" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Report on GitHub" })).toBeTruthy();
  });

  it("wears the bug icon in its heading, not a stethoscope", () => {
    const { container } = renderPage();
    expect(container.querySelector("header .lucide-bug")).not.toBeNull();
    expect(container.querySelector(".lucide-stethoscope")).toBeNull();
  });

  it("marks the Diagnostics visit on mount, clearing the bug icon's counter (#485)", () => {
    expect(window.localStorage.getItem(LAST_VISIT_STORAGE_KEY)).toBeNull();
    const before = Date.now();
    renderPage();
    expect(getLastDiagnosticsVisit()).toBeGreaterThanOrEqual(before);
  });
});
