// @vitest-environment jsdom
/**
 * The content of the Diagnostics page.
 *
 * The test that earns this file is "the backend is dead". A panel that goes
 * blank when the backend does not answer is useless exactly when someone needs
 * it, so the panel must still show the app's own version, platform and log
 * path, still show the app-log and session errors, and say plainly that the
 * backend part is missing.
 */
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

const { getMock } = vi.hoisted(() => ({ getMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: getMock },
  apiClient: { get: getMock },
}));

import DiagnosticsPanel from "./DiagnosticsPanel";
import { recordSessionError, resetSessionErrors } from "../utils/errorCapture";
import i18n from "../i18n";

const BACKEND = {
  environment: {
    platform: "Darwin",
    platform_release: "25.5.0",
    architecture: "arm64",
    python_version: "3.12.9",
    engine: "MLX_Engine",
    loaded_model: "mlx-community/Qwen3-4B-4bit",
    loaded_model_id: 42,
    cpu_model: "Apple M3 Pro",
    gpu_name: "Apple M3 Pro GPU",
    compute_capability: null,
    vram_total_gb: null,
    db: "ok",
    backend_log_path: "/Users/x/Library/Logs/erudi/backend.log",
  },
  recent_errors: [
    {
      timestamp: "2026-09-05T10:00:00.000Z",
      level: "ERROR",
      request_id: "be-1f2e3d4c",
      message: "engine refused to start",
    },
  ],
};

let appLogTail;
let revealLog;

beforeEach(() => {
  resetSessionErrors();
  getMock.mockResolvedValue(BACKEND);
  // A record of the main process's own: shown whether or not the backend
  // answers. (An echo of the backend's stdout would be dropped while the
  // backend answers, since backend.log holds the same record.)
  appLogTail = vi.fn().mockResolvedValue([
    {
      timestamp: "2026-09-05T09:00:00.000Z",
      level: "WARNING",
      message: "[main] WARN slow boot",
    },
  ]);
  revealLog = vi.fn().mockResolvedValue({ success: true });
  window.diagnosticsAPI = {
    getAppInfo: vi.fn().mockResolvedValue({
      version: "1.0.0",
      platform: "darwin",
      arch: "arm64",
      electron: "38.0.0",
      appLogPath: "/tmp/erudi-backend.log",
    }),
    appLogTail,
    revealLog,
  };
  Object.defineProperty(window.navigator, "clipboard", {
    value: { writeText: vi.fn().mockResolvedValue(undefined) },
    configurable: true,
  });
});

afterEach(async () => {
  cleanup();
  delete window.diagnosticsAPI;
  vi.clearAllMocks();
  await i18n.changeLanguage("en");
});

describe("DiagnosticsPanel — backend up", () => {
  it("shows the environment summary from both sides", async () => {
    render(<DiagnosticsPanel />);
    expect(await screen.findByText("MLX_Engine")).toBeTruthy();
    expect(screen.getByText("1.0.0")).toBeTruthy();
    expect(screen.getByText("mlx-community/Qwen3-4B-4bit")).toBeTruthy();
    expect(screen.getByText("3.12.9")).toBeTruthy();
    expect(screen.getByText("/Users/x/Library/Logs/erudi/backend.log")).toBeTruthy();
    expect(screen.getByText("/tmp/erudi-backend.log")).toBeTruthy();
  });

  it("merges backend, app-log and session errors into one list", async () => {
    recordSessionError({ origin: "window.onerror", message: "render blew up" });
    render(<DiagnosticsPanel />);
    // Each message shows twice: once in the list, once inside the copyable
    // block, which is the point of the block.
    expect(await screen.findAllByText(/engine refused to start/)).toHaveLength(2);
    expect(screen.getAllByText(/slow boot/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/render blew up/).length).toBeGreaterThan(0);
  });

  it("warns next to the copy block that logs can contain conversation content", async () => {
    render(<DiagnosticsPanel />);
    const note = await screen.findByText(/Logs can contain the text of your conversations/);
    const block = screen.getByLabelText("Diagnostics to copy").parentElement;
    expect(block.contains(note)).toBe(true);
  });

  it("lists the entries rather than the quiet state when there is something to show", async () => {
    render(<DiagnosticsPanel />);
    expect(await screen.findAllByText(/engine refused to start/)).toHaveLength(2);
    expect(screen.queryByText("No warning or error recorded.")).toBeNull();
    expect(document.querySelector(".lucide-circle-check")).toBeNull();
  });

  it("puts the whole report in the copyable block", async () => {
    render(<DiagnosticsPanel />);
    const area = await screen.findByLabelText("Diagnostics to copy");
    expect(area.value).toContain("Erudi 1.0.0");
    expect(area.value).toContain("MLX_Engine");
    expect(area.value).toContain("engine refused to start");
    expect(area.value).toContain("be-1f2e3d4c");
  });

  it("reveals the log folder through the bridge", async () => {
    render(<DiagnosticsPanel />);
    fireEvent.click(await screen.findByRole("button", { name: "Open log folder" }));
    await waitFor(() => expect(revealLog).toHaveBeenCalled());
  });
});

describe("DiagnosticsPanel — backend down", () => {
  beforeEach(() => {
    getMock.mockRejectedValue(new Error("ECONNREFUSED"));
  });

  it("still shows what the app knows about itself", async () => {
    render(<DiagnosticsPanel />);
    expect(await screen.findByText("The backend did not answer")).toBeTruthy();
    expect(screen.getByText("1.0.0")).toBeTruthy();
    expect(screen.getByText("/tmp/erudi-backend.log")).toBeTruthy();
  });

  it("still shows the app-side errors", async () => {
    recordSessionError({ origin: "window.onerror", message: "render blew up" });
    render(<DiagnosticsPanel />);
    expect((await screen.findAllByText(/slow boot/)).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/render blew up/).length).toBeGreaterThan(0);
  });

  it("still offers a copyable report and the GitHub route", async () => {
    render(<DiagnosticsPanel />);
    // The textarea exists from the first render; its text fills in once the
    // three sources have settled, so wait for the content, not the element.
    const area = await screen.findByLabelText("Diagnostics to copy");
    await waitFor(() => expect(area.value).toContain("Erudi 1.0.0"));
    expect(area.value).toContain("backend did not answer");
    expect(screen.getByRole("button", { name: "Report on GitHub" })).toBeTruthy();
  });
});

describe("DiagnosticsPanel — nothing recorded", () => {
  beforeEach(() => {
    getMock.mockResolvedValue({ ...BACKEND, recent_errors: [] });
    appLogTail.mockResolvedValue([]);
  });

  it("says so in one quiet line, with nothing else in the errors area", async () => {
    render(<DiagnosticsPanel />);
    const empty = await screen.findByText("No warning or error recorded.");
    // A check mark, not a warning: nothing happened and nothing is asked.
    expect(empty.parentElement.querySelector(".lucide-circle-check")).not.toBeNull();
    expect(document.querySelector("ul")).toBeNull();
    expect(document.querySelector("pre")).toBeNull();
  });

  it("keeps the log privacy note with the copy block, where it still applies", async () => {
    render(<DiagnosticsPanel />);
    const note = await screen.findByText(/Logs can contain the text of your conversations/);
    const block = screen.getByLabelText("Diagnostics to copy").parentElement;
    expect(block.contains(note)).toBe(true);
  });

  it("still offers the report route and the log folder", async () => {
    render(<DiagnosticsPanel />);
    expect(await screen.findByText("Report a problem")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Report on GitHub" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Open log folder" })).toBeTruthy();
  });
});

describe("DiagnosticsPanel — no Electron bridge", () => {
  it("renders from the backend alone rather than throwing", async () => {
    delete window.diagnosticsAPI;
    render(<DiagnosticsPanel />);
    expect(await screen.findByText("MLX_Engine")).toBeTruthy();
  });
});
