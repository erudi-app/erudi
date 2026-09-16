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

  it("shows an uncaught renderer error once: the file record, with the session's count", () => {
    // The capture writes the error to the app log through the bridge AND keeps
    // it in the session buffer. The file record survives a reload, so it is
    // the one shown; the session entry only contributes its repeat count.
    const entries = mergeRecentErrors({
      backend: null,
      appLog: [
        {
          timestamp: "2026-09-05T11:00:00.120Z",
          level: "ERROR",
          message:
            "[renderer:renderer:uncaught] ERROR window.onerror: render blew up Error: render blew up\n    at x",
        },
      ],
      sessionErrors: [
        {
          timestamp: "2026-09-05T11:00:00.000Z",
          origin: "window.onerror",
          message: "render blew up",
          stack: "Error: render blew up\n    at x",
          count: 3,
        },
      ],
    });
    expect(entries).toHaveLength(1);
    expect(entries[0].source).toBe("app");
    expect(entries[0].count).toBe(3);
  });

  it("keeps a session error the app log does not have", () => {
    // The bridge was absent (a browser, a test, an early boot): the session
    // buffer is the only copy.
    const entries = mergeRecentErrors({
      backend: null,
      appLog: [
        {
          timestamp: "2026-09-05T11:00:00.120Z",
          level: "ERROR",
          message: "[renderer:renderer:uncaught] ERROR window.onerror: a different error",
        },
      ],
      sessionErrors: [
        { timestamp: "2026-09-05T11:00:00.000Z", origin: "unhandledrejection", message: "lost" },
      ],
    });
    expect(entries.map((e) => e.source)).toEqual(["session", "app"]);
  });

  it("drops the app log's echo of backend output when the backend answered", () => {
    // main.js copies every backend stdout line into the app log; the backend's
    // own log has the same record with its traceback attached.
    const entries = mergeRecentErrors({
      backend: BACKEND,
      appLog: [
        {
          timestamp: "2026-09-05T10:00:00.010Z",
          level: "ERROR",
          message:
            "Backend stdout: [ERROR] 2026-09-05T10:00:00.000Z [be-1] - erudi - x.py:1 - boom",
        },
        { timestamp: "2026-09-05T10:00:01.000Z", level: "ERROR", message: "[main] ERROR kept" },
      ],
      sessionErrors: [],
    });
    expect(entries.map((e) => [e.source, e.message])).toEqual([
      ["backend", "boom"],
      ["app", "[main] ERROR kept"],
    ]);
  });

  it("keeps the echo of backend output when the backend did not answer", () => {
    // A backend that will not start leaves its last words only in the echo.
    const entries = mergeRecentErrors({
      backend: null,
      appLog: [
        {
          timestamp: "2026-09-05T10:00:00.010Z",
          level: "ERROR",
          message: "Backend stderr: [ERROR] Startup failed: alembic exploded",
        },
      ],
      sessionErrors: [],
    });
    expect(entries).toHaveLength(1);
    expect(entries[0].message).toContain("alembic exploded");
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

  it("drops an environmental ERROR (#485): an offline HuggingFace download never appears", () => {
    const entries = mergeRecentErrors({
      backend: {
        recent_errors: [
          {
            timestamp: "2026-09-05T10:00:00.000Z",
            level: "ERROR",
            request_id: "be-9",
            message:
              "POST /erudi/llms/download -> 503 HUGGINGFACE_API_ERROR: you appear to be offline - check your connection and retry",
          },
          {
            timestamp: "2026-09-05T10:00:01.000Z",
            level: "ERROR",
            request_id: "be-10",
            message: "GET /erudi/conversations -> 500 DATABASE_ERROR: a real bug",
          },
        ],
      },
      appLog: [],
      sessionErrors: [],
    });
    expect(entries.map((e) => e.message)).toEqual([
      "GET /erudi/conversations -> 500 DATABASE_ERROR: a real bug",
    ]);
  });

  it("keeps an environmental WARNING: the filter never touches WARNING/INFO-adjacent levels", () => {
    const entries = mergeRecentErrors({
      backend: null,
      appLog: [
        {
          timestamp: "2026-09-05T10:00:00.000Z",
          level: "WARNING",
          message: "you appear to be offline - retrying later",
        },
      ],
      sessionErrors: [],
    });
    expect(entries).toHaveLength(1);
  });

  it("folds identical backend records into one entry with a repeat count (#534)", () => {
    // Observed on the 1.1.2 pass: the same Hugging Face advice line three
    // times in one session, listed as three rows and three lines of the
    // copied report.
    const repeated =
      "Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed.";
    const entries = mergeRecentErrors({
      backend: {
        recent_errors: [
          {
            timestamp: "2026-09-05T10:00:00.000Z",
            level: "WARNING",
            request_id: "be-1",
            message: repeated,
          },
          {
            timestamp: "2026-09-05T10:00:01.000Z",
            level: "WARNING",
            request_id: "be-2",
            message: repeated,
          },
          {
            timestamp: "2026-09-05T10:00:02.000Z",
            level: "WARNING",
            request_id: "be-3",
            message: repeated,
          },
        ],
      },
      appLog: [],
      sessionErrors: [],
    });
    expect(entries).toHaveLength(1);
    expect(entries[0].count).toBe(3);
    // The newest occurrence names the entry: "when did this last happen" is
    // the question, and the request id has to point at the same turn.
    expect(entries[0].timestamp).toBe("2026-09-05T10:00:02.000Z");
    expect(entries[0].requestId).toBe("be-3");
  });

  it("folds only within a source and a level, never across them", () => {
    const same = "identical text";
    const entries = mergeRecentErrors({
      backend: {
        recent_errors: [
          {
            timestamp: "2026-09-05T10:00:00.000Z",
            level: "WARNING",
            request_id: "be-1",
            message: same,
          },
          {
            timestamp: "2026-09-05T10:00:01.000Z",
            level: "ERROR",
            request_id: "be-2",
            message: same,
          },
        ],
      },
      // Not a "Backend stdout:" echo, so it survives the echo filter and
      // stands as an app record of its own.
      appLog: [{ timestamp: "2026-09-05T10:00:02.000Z", level: "WARNING", message: same }],
      sessionErrors: [],
    });
    expect(entries).toHaveLength(3);
    expect(entries.every((e) => e.count === 1)).toBe(true);
  });

  it("keeps a single record untouched, count and request id included", () => {
    const entries = mergeRecentErrors({ backend: BACKEND, appLog: [], sessionErrors: [] });
    expect(entries).toHaveLength(1);
    expect(entries[0].count).toBe(1);
    expect(entries[0].requestId).toBe("be-1");
  });

  it("folds repeats before the cap, so folding frees slots", () => {
    const noisy = Array.from({ length: 6 }, (_, i) => ({
      timestamp: `2026-09-05T10:00:0${i}.000Z`,
      level: "WARNING",
      request_id: `be-${i}`,
      message: "the same noisy line",
    }));
    const entries = mergeRecentErrors({
      backend: {
        recent_errors: [
          {
            timestamp: "2026-09-05T09:00:00.000Z",
            level: "ERROR",
            request_id: "be-x",
            message: "a real error",
          },
          ...noisy,
        ],
      },
      appLog: [],
      sessionErrors: [],
      limit: 2,
    });
    // Without folding the six repeats would fill the cap and push the real
    // error out; folded, both fit.
    expect(entries).toHaveLength(2);
    expect(entries[0].message).toBe("a real error");
    expect(entries[1].count).toBe(6);
  });

  it("keeps two different bugs apart when they throw the same text", () => {
    // A session entry keeps its stack outside the message, so the message
    // alone cannot tell two components' identical TypeErrors apart. Folding
    // them would sum their counts under the first stack and lose the second
    // bug entirely.
    const entries = mergeRecentErrors({
      backend: null,
      appLog: [],
      sessionErrors: [
        {
          timestamp: "2026-09-05T11:00:00.000Z",
          origin: "window.onerror",
          message: "Cannot read properties of undefined (reading 'x')",
          stack: "TypeError: ...\n    at ChatPage",
          count: 2,
        },
        {
          timestamp: "2026-09-05T11:05:00.000Z",
          origin: "window.onerror",
          message: "Cannot read properties of undefined (reading 'x')",
          stack: "TypeError: ...\n    at ArenaPage",
          count: 3,
        },
      ],
    });
    expect(entries).toHaveLength(2);
    const byStack = Object.fromEntries(entries.map((e) => [e.stack, e]));
    expect(byStack["TypeError: ...\n    at ChatPage"].count).toBe(2);
    expect(byStack["TypeError: ...\n    at ArenaPage"].count).toBe(3);
    expect(byStack["TypeError: ...\n    at ArenaPage"].timestamp).toBe("2026-09-05T11:05:00.000Z");
  });

  it("sums the members' own counts, not the number of members", () => {
    // Every member already carries a count above 1, so a fold that counted
    // rows (2) instead of summing (5) would be caught here.
    const same = {
      origin: "window.onerror",
      message: "render loop",
      stack: "Error: render loop\n    at Widget",
    };
    const entries = mergeRecentErrors({
      backend: null,
      appLog: [],
      sessionErrors: [
        { ...same, timestamp: "2026-09-05T11:00:00.000Z", count: 2 },
        { ...same, timestamp: "2026-09-05T11:01:00.000Z", count: 3 },
      ],
    });
    expect(entries).toHaveLength(1);
    expect(entries[0].count).toBe(5);
    expect(entries[0].timestamp).toBe("2026-09-05T11:01:00.000Z");
    expect(entries[0].stack).toBe("Error: render loop\n    at Widget");
  });

  it("keeps the session buffer's count when its app-log twin repeats", () => {
    // The session entry folds into the file record first (Math.max), then the
    // fold runs: the surviving count must still carry what the session knew.
    const message =
      "[renderer:renderer:uncaught] ERROR window.onerror: render blew up Error: render blew up";
    const entries = mergeRecentErrors({
      backend: null,
      appLog: [{ timestamp: "2026-09-05T11:00:00.120Z", level: "ERROR", message }],
      sessionErrors: [
        {
          timestamp: "2026-09-05T11:00:00.000Z",
          origin: "window.onerror",
          message: "render blew up",
          stack: "Error: render blew up\n    at x",
          count: 4,
        },
      ],
    });
    expect(entries).toHaveLength(1);
    expect(entries[0].count).toBe(4);
    expect(entries[0].source).toBe("app");
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

  it("rounds the VRAM the driver reports at full precision", () => {
    const cuda = {
      environment: {
        cpu_model: "AMD Ryzen 7 7700 8-Core Processor",
        gpu_name: "NVIDIA GeForce RTX 5060 Ti",
        // What the driver actually hands back.
        vram_total_gb: 15.9287109375,
        compute_capability: "12.0",
      },
    };
    expect(buildPrefill({ app: APP, backend: cuda }).hardware).toContain("15.9 GB VRAM");
    expect(buildPrefill({ app: APP, backend: cuda }).hardware).not.toContain("15.9287109375");
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
