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
let clipboardWriteText;

beforeEach(() => {
  resetSessionErrors();
  getMock.mockResolvedValue(BACKEND);
  appLogTail = vi.fn().mockResolvedValue([
    {
      timestamp: "2026-09-05T09:00:00.000Z",
      level: "WARNING",
      message: "Backend stdout: [WARNING] slow boot",
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
  clipboardWriteText = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(window.navigator, "clipboard", {
    value: { writeText: clipboardWriteText },
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
    // Every value comes from the same state-settling render, but wait on all
    // of them together rather than trusting that the first one to appear
    // means the rest already landed.
    await waitFor(() => {
      expect(screen.getByText("MLX_Engine")).toBeTruthy();
      expect(screen.getByText("1.0.0")).toBeTruthy();
      expect(screen.getByText("mlx-community/Qwen3-4B-4bit")).toBeTruthy();
      expect(screen.getByText("3.12.9")).toBeTruthy();
    });
  });

  it("does not show the log-path rows, nor a text preview", async () => {
    render(<DiagnosticsPanel />);
    await waitFor(() => expect(screen.getByText("MLX_Engine")).toBeTruthy());
    expect(screen.queryByText("/Users/x/Library/Logs/erudi/backend.log")).toBeNull();
    expect(screen.queryByText("/tmp/erudi-backend.log")).toBeNull();
    expect(document.querySelector("textarea")).toBeNull();
  });

  it("merges backend, app-log and session errors into one list", async () => {
    recordSessionError({ origin: "window.onerror", message: "render blew up" });
    render(<DiagnosticsPanel />);
    await waitFor(() => {
      expect(screen.getAllByText(/engine refused to start/)).toHaveLength(1);
      expect(screen.getAllByText(/slow boot/).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/render blew up/).length).toBeGreaterThan(0);
    });
  });

  it("lists the entries rather than the quiet state when there is something to show", async () => {
    render(<DiagnosticsPanel />);
    await waitFor(() => expect(screen.getAllByText(/engine refused to start/)).toHaveLength(1));
    expect(screen.queryByText("No warning or error recorded.")).toBeNull();
    expect(document.querySelector(".lucide-circle-check")).toBeNull();
  });

  it("copies the whole report, including the log paths, through a single button", async () => {
    render(<DiagnosticsPanel />);
    // The copy button only mounts once `entries` reflects the loaded data, so
    // waiting for it also waits for `app` and `backend` to have settled —
    // unlike the old textarea, which existed from the very first render.
    const copyButton = await screen.findByRole("button", { name: "Copy the full report" });
    fireEvent.click(copyButton);
    await waitFor(() => expect(clipboardWriteText).toHaveBeenCalled());
    const copiedText = clipboardWriteText.mock.calls[0][0];
    expect(copiedText).toContain("Erudi 1.0.0");
    expect(copiedText).toContain("MLX_Engine");
    expect(copiedText).toContain("engine refused to start");
    expect(copiedText).toContain("be-1f2e3d4c");
    expect(copiedText).toContain("/Users/x/Library/Logs/erudi/backend.log");
    expect(copiedText).toContain("/tmp/erudi-backend.log");
  });

  it("offers the GitHub report route and the contact page alongside the copy button", async () => {
    render(<DiagnosticsPanel />);
    await screen.findByRole("button", { name: "Copy the full report" });
    expect(screen.getByRole("button", { name: "Report on GitHub" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "Write to us on the contact page" })).toBeTruthy();
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
    await waitFor(() => {
      expect(screen.getByText("The backend did not answer")).toBeTruthy();
      expect(screen.getByText("1.0.0")).toBeTruthy();
    });
  });

  it("still shows the app-side errors", async () => {
    recordSessionError({ origin: "window.onerror", message: "render blew up" });
    render(<DiagnosticsPanel />);
    await waitFor(() => {
      expect(screen.getAllByText(/slow boot/).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/render blew up/).length).toBeGreaterThan(0);
    });
  });

  it("still offers a copyable report and the GitHub route", async () => {
    render(<DiagnosticsPanel />);
    // Wait on the copy button itself (mounted only once `entries` reflects
    // the loaded data) rather than an element present from the first render:
    // reading a still-empty value right after the element appears is exactly
    // what made the old textarea-based version of this test flaky.
    const copyButton = await screen.findByRole("button", { name: "Copy the full report" });
    fireEvent.click(copyButton);
    await waitFor(() => expect(clipboardWriteText).toHaveBeenCalled());
    const copiedText = clipboardWriteText.mock.calls[0][0];
    expect(copiedText).toContain("Erudi 1.0.0");
    expect(copiedText).toContain("backend did not answer");
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

  it("shows no copy button and nothing about handling errors", async () => {
    render(<DiagnosticsPanel />);
    await screen.findByText("No warning or error recorded.");
    expect(screen.queryByRole("button", { name: /copy/i })).toBeNull();
    expect(screen.queryByText(/Paste the report/)).toBeNull();
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
