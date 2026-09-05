/**
 * Merging the three error sources and formatting the text a user copies.
 *
 * The case that decides whether this feature is worth anything is the one
 * where the backend is dead: the report must still describe the machine and
 * still carry the app-side errors, and it must say the backend part is
 * missing rather than pretend it is empty.
 */
import { describe, it, expect } from "vitest";

import {
  buildPrefill,
  formatDiagnosticsText,
  mergeRecentErrors,
  osLabel,
} from "./diagnosticsReport";

const APP = {
  version: "1.0.0",
  platform: "darwin",
  arch: "arm64",
  electron: "38.0.0",
  appLogPath: "/tmp/erudi-backend.log",
};

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
    { timestamp: "2026-09-05T10:00:00.000Z", level: "ERROR", request_id: "be-1", message: "boom" },
  ],
};

describe("osLabel", () => {
  it("maps a platform and architecture to a bug-report option", () => {
    expect(osLabel("darwin", "arm64")).toBe("macOS (Apple Silicon)");
    expect(osLabel("darwin", "x64")).toBe("macOS (Intel)");
    expect(osLabel("win32", "x64")).toBe("Windows 10 / 11");
    expect(osLabel("linux", "x64")).toBe("Linux");
  });

  it("returns null for something it does not recognise", () => {
    expect(osLabel("freebsd", "x64")).toBeNull();
    expect(osLabel(undefined, undefined)).toBeNull();
  });
});

describe("mergeRecentErrors", () => {
  it("merges the three sources into one timeline, oldest first", () => {
    const entries = mergeRecentErrors({
      backend: BACKEND,
      appLog: [{ timestamp: "2026-09-05T09:00:00.000Z", level: "WARNING", message: "app warn" }],
      sessionErrors: [
        {
          timestamp: "2026-09-05T11:00:00.000Z",
          origin: "window.onerror",
          message: "render blew up",
          stack: "at x",
          count: 3,
        },
      ],
    });
    expect(entries.map((e) => e.source)).toEqual(["app", "backend", "session"]);
    // A session entry keeps its origin in front of the message: "which
    // handler caught this" is half of what makes it readable in a report.
    expect(entries.map((e) => e.message)).toEqual([
      "app warn",
      "boom",
      "window.onerror: render blew up",
    ]);
    expect(entries[2].count).toBe(3);
    expect(entries[1].requestId).toBe("be-1");
  });

  it("keeps the app-side sources when the backend is missing", () => {
    const entries = mergeRecentErrors({
      backend: null,
      appLog: [{ timestamp: "2026-09-05T09:00:00.000Z", level: "ERROR", message: "spawn failed" }],
      sessionErrors: [],
    });
    expect(entries).toHaveLength(1);
    expect(entries[0].source).toBe("app");
  });

  it("caps the merged list and keeps the newest entries", () => {
    const appLog = Array.from({ length: 400 }, (_, i) => ({
      timestamp: `2026-09-05T${String(i % 24).padStart(2, "0")}:00:00.000Z`,
      level: "ERROR",
      message: `e${i}`,
    }));
    expect(mergeRecentErrors({ backend: null, appLog, sessionErrors: [], limit: 50 })).toHaveLength(
      50
    );
  });

  it("survives every source being absent", () => {
    expect(mergeRecentErrors({})).toEqual([]);
  });
});

describe("buildPrefill", () => {
  it("fills the issue form's text fields from both sides", () => {
    expect(buildPrefill({ app: APP, backend: BACKEND })).toEqual({
      version: "1.0.0",
      os: "macOS (Apple Silicon)",
      hardware: "Apple M3 Pro / Apple M3 Pro GPU",
      model: "mlx-community/Qwen3-4B-4bit",
    });
  });

  it("still names the version and the OS when the backend is down", () => {
    const prefill = buildPrefill({ app: APP, backend: null });
    expect(prefill.version).toBe("1.0.0");
    expect(prefill.os).toBe("macOS (Apple Silicon)");
    expect(prefill.hardware).toBeNull();
    expect(prefill.model).toBeNull();
  });

  it("takes an explicit hardware string over the backend's", () => {
    // The engine-failure dialog knows the card that failed from the notice
    // itself, and opens when the backend has nothing useful to say about it.
    const prefill = buildPrefill({
      app: APP,
      backend: BACKEND,
      hardware: "NVIDIA GeForce GTX 1080, compute 6.1",
    });
    expect(prefill.hardware).toBe("NVIDIA GeForce GTX 1080, compute 6.1");
    expect(prefill.version).toBe("1.0.0");
  });

  it("falls back to the backend's hardware when the explicit one is empty", () => {
    expect(buildPrefill({ app: APP, backend: BACKEND, hardware: "" }).hardware).toBe(
      "Apple M3 Pro / Apple M3 Pro GPU"
    );
  });

  it("reports VRAM when the engine is CUDA", () => {
    const cuda = {
      environment: {
        ...BACKEND.environment,
        engine: "CUDA_Engine",
        cpu_model: "AMD Ryzen 7",
        gpu_name: "NVIDIA RTX 4070",
        vram_total_gb: 12,
        compute_capability: "8.9",
      },
    };
    expect(buildPrefill({ app: APP, backend: cuda }).hardware).toBe(
      "AMD Ryzen 7 / NVIDIA RTX 4070, 12 GB VRAM, compute 8.9"
    );
  });
});

describe("formatDiagnosticsText", () => {
  it("names the version, the engine, the model and the log paths", () => {
    const text = formatDiagnosticsText({
      app: APP,
      backend: BACKEND,
      entries: mergeRecentErrors({ backend: BACKEND }),
    });
    expect(text).toContain("Erudi 1.0.0");
    expect(text).toContain("MLX_Engine");
    expect(text).toContain("mlx-community/Qwen3-4B-4bit");
    expect(text).toContain("/tmp/erudi-backend.log");
    expect(text).toContain("/Users/x/Library/Logs/erudi/backend.log");
    expect(text).toContain("[ERROR]");
    expect(text).toContain("boom");
    expect(text).toContain("be-1");
  });

  it("says the backend part is missing instead of leaving a blank", () => {
    const text = formatDiagnosticsText({ app: APP, backend: null, entries: [] });
    expect(text).toContain("Erudi 1.0.0");
    expect(text).toContain("backend did not answer");
    // It must not claim there were no errors when it simply could not look.
    expect(text).not.toContain("MLX_Engine");
  });

  it("marks a repeated error with its count", () => {
    const text = formatDiagnosticsText({
      app: APP,
      backend: null,
      entries: mergeRecentErrors({
        sessionErrors: [
          {
            timestamp: "2026-09-05T11:00:00.000Z",
            origin: "window.onerror",
            message: "loop",
            count: 412,
          },
        ],
      }),
    });
    expect(text).toContain("x412");
  });

  it("says so when there is nothing to report", () => {
    const text = formatDiagnosticsText({ app: APP, backend: BACKEND, entries: [] });
    expect(text).toContain("no warnings or errors");
  });

  it("produces a plain string even with nothing at all", () => {
    expect(typeof formatDiagnosticsText({})).toBe("string");
  });
});
