// @vitest-environment jsdom
/**
 * The bug-icon counter's decision logic (#485): which merged Diagnostics
 * records are real defects, which of those are new, and how the badge reads.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

import {
  isEnvironmental,
  isHiddenFromDiagnostics,
  isCountableError,
  countNewErrors,
  badgeLabel,
  BADGE_CAP,
  LAST_VISIT_STORAGE_KEY,
  getLastDiagnosticsVisit,
  markDiagnosticsVisited,
  subscribeDiagnosticsVisit,
} from "./bugCounter";

describe("isEnvironmental", () => {
  it("matches the HuggingFace model-download offline message", () => {
    // backend.log record shape: "POST /erudi/llms/download -> 503
    // HUGGINGFACE_API_ERROR: you appear to be offline - check your
    // connection and retry"
    expect(
      isEnvironmental({
        message:
          "POST /erudi/llms/download -> 503 HUGGINGFACE_API_ERROR: you appear to be offline - check your connection and retry",
      })
    ).toBe(true);
  });

  it("matches the renderer-side echo of the same offline download failure", () => {
    // frontend api.failure / DownloadModalContext record shape.
    expect(
      isEnvironmental({
        message:
          'Download job 7 failed with status "failed" {"error":"you appear to be offline - check your connection and retry"}',
      })
    ).toBe(true);
  });

  it("matches the embedding model download failing offline", () => {
    expect(
      isEnvironmental({
        message:
          "Embedding model download failed: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded",
      })
    ).toBe(true);
    expect(
      isEnvironmental({
        message: "Embedding model download failed: [Errno 8] nodename nor servname provided",
      })
    ).toBe(true);
  });

  it("does not treat an unrelated failure that shares a network marker as environmental", () => {
    // A local Postgres failure can legitimately say "connection error" in its
    // trace; it must not be swallowed by the embedding-model scope, which
    // requires the "embedding model download failed" prefix too.
    expect(
      isEnvironmental({
        message: "PUT /erudi/conversations/1 -> 500 DATABASE_ERROR: Could not update conversation",
      })
    ).toBe(false);
  });

  it("does not treat a real HuggingFace API failure (not offline) as environmental", () => {
    // backend/src/domains/llms/services.py's other HuggingFaceAPIException
    // raise site: a split GGUF quant missing parts. A real defect.
    expect(
      isEnvironmental({
        message:
          "POST /erudi/llms/download -> 503 HUGGINGFACE_API_ERROR: model.gguf is part of a 3-part split quant, but the repository listing only exposes 2 of them",
      })
    ).toBe(false);
  });

  it("matches a fetch failure toward huggingface.co or github.com (defense in depth)", () => {
    expect(
      isEnvironmental({ message: "api.failure Failed to fetch https://huggingface.co/api/x" })
    ).toBe(true);
    expect(isEnvironmental({ message: "request to github.com timed out after 30000ms" })).toBe(
      true
    );
  });

  it("does not treat a request that failed against our own local backend as environmental", () => {
    expect(
      isEnvironmental({
        message: "GET /erudi/conversations -> 500 INTERNAL_SERVER_ERROR: unexpected failure",
      })
    ).toBe(false);
  });

  it("is false for an entry with no message", () => {
    expect(isEnvironmental({})).toBe(false);
    expect(isEnvironmental({ message: "" })).toBe(false);
  });

  it("matches a gated HuggingFace repo downloaded with no token", () => {
    // backend/src/domains/llms/services.py:798-805 -> UnsupportedPlatformException,
    // logged by endpoints.py's `_run_download_task` as "... not supported on
    // this platform: requires HuggingFace authentication and cannot be
    // downloaded anonymously".
    expect(
      isEnvironmental({
        message:
          "Download job 9 failed for LLM 3 (meta-llama/Llama-Guard): meta-llama/Llama-Guard not supported on this platform: requires HuggingFace authentication and cannot be downloaded anonymously",
      })
    ).toBe(true);
  });

  it("does not treat an unsupported-platform failure for an unrelated reason as environmental", () => {
    // Same exception class, different `reason` text (e.g. CUDA unavailable) --
    // must not be swept in by a loose match on "not supported on this platform".
    expect(
      isEnvironmental({
        message: "CUDA not supported on this platform: no NVIDIA GPU detected",
      })
    ).toBe(false);
  });

  it("matches HuggingFace rate-limiting or a HuggingFace-side 5xx during a download", () => {
    // backend/src/domains/llms/services.py:805 bare re-raises HfHubHTTPError
    // when it is neither gated nor offline; requests/huggingface_hub format
    // it as "<status> <reason> Error: ... for url: https://huggingface.co/...".
    expect(
      isEnvironmental({
        message:
          "Download job 4 failed for LLM 2 (org/model): 429 Client Error: Too Many Requests for url: https://huggingface.co/api/models/org/model",
      })
    ).toBe(true);
    expect(
      isEnvironmental({
        message:
          "Download job 5 failed for LLM 2 (org/model): 503 Server Error: Service Unavailable for url: https://huggingface.co/org/model/resolve/main/model.gguf",
      })
    ).toBe(true);
  });

  it("does not treat our own local 429/503 as environmental just because a status code matches", () => {
    // No mention of huggingface.co: this is our own backend.
    expect(
      isEnvironmental({
        message: "GET /erudi/health -> 503 SERVICE_UNAVAILABLE: database is recovering",
      })
    ).toBe(false);
  });

  it("matches a disk-full failure during a download or KB ingestion", () => {
    // backend/src/domains/llms/endpoints.py's `_run_download_task` and
    // backend/src/domains/knowledge_base/services.py's `_ingest_one_file`
    // both log a bare OSError at ERROR through their generic exception
    // handler; ENOSPC/EDQUOT surface with Python's structured errno text.
    expect(
      isEnvironmental({
        message: "KB 2: ingestion failed for report.pdf: [Errno 28] No space left on device",
      })
    ).toBe(true);
    expect(
      isEnvironmental({
        message:
          "Download job 6 failed for LLM 1 (org/model): [WinError 112] There is not enough space on the disk",
      })
    ).toBe(true);
    expect(
      isEnvironmental({
        message: "KB 4: ingestion failed for report.pdf: [Errno 122] Disk quota exceeded",
      })
    ).toBe(true);
  });

  it("does NOT hide a real parser crash just because the user's own file name happens to say a disk is full (adversarial, #485 review)", () => {
    // KB ingestion messages embed the file's name verbatim
    // ("_ingest_one_file: ingestion failed for <file.name>: <exc>"). A file a
    // user genuinely might have -- an error screenshot, a saved support doc
    // -- named exactly with the English disk-full wording must not hide an
    // unrelated real defect behind that coincidence. Only the STRUCTURED
    // errno/winerror segment of the actual exception text is trusted, never
    // an arbitrary substring of the whole display message.
    expect(
      isEnvironmental({
        message:
          "KB 3: ingestion failed for No space left on device.pdf: UnicodeDecodeError: invalid start byte",
      })
    ).toBe(false);
    expect(
      isEnvironmental({
        message:
          "KB 5: ingestion failed for There is not enough space on the disk.docx: zipfile.BadZipFile: File is not a zip file",
      })
    ).toBe(false);
    expect(
      isEnvironmental({
        message: "KB 6: ingestion failed for disk quota exceeded notes.txt: ValueError: bad header",
      })
    ).toBe(false);
  });

  it("matches the backend startup failing because every port it scans is already taken", () => {
    // backend/run.py's log_startup_failure("NO_PORT_AVAILABLE", ...), echoed
    // into erudi-backend.log even when the backend itself never came up.
    expect(
      isEnvironmental({
        message:
          "startup_error NO_PORT_AVAILABLE: Ports 27182-27199 all busy, failed to free 27190",
      })
    ).toBe(true);
  });

  it("does not treat a sibling startup_error code as environmental", () => {
    // IMPORT_ERROR/DATA_PREP_ERROR/CRASH_BEFORE_READY/UNEXPECTED_ERROR/
    // POLLING_ERROR are internal failures, not the machine's fault.
    expect(
      isEnvironmental({
        message:
          "startup_error IMPORT_ERROR: Failed to import FastAPI application: ModuleNotFoundError",
      })
    ).toBe(false);
    expect(
      isEnvironmental({
        message: "startup_error CRASH_BEFORE_READY: Backend thread exited before binding port",
      })
    ).toBe(false);
  });

  it("does not treat a dead local inference child as environmental despite mentioning a connection", () => {
    // backend/src/agents/runner.py's "Agent streaming failed" record and
    // base_chat_server_engine.py's EngineException text both describe OUR
    // OWN child process against 127.0.0.1, never huggingface.co/github.com.
    expect(
      isEnvironmental({
        message:
          "Agent streaming failed: llm=3 (Qwen2.5-0.5B), thread_id=abc; llama-server child is dead (exit code -9)",
      })
    ).toBe(false);
    expect(
      isEnvironmental({
        message:
          "llama-server chat-completions probe failed (pid 123, port 27200, exit=None): ConnectionError: Max retries exceeded with url: /v1/chat/completions",
      })
    ).toBe(false);
  });
});

describe("isHiddenFromDiagnostics", () => {
  it("hides an environmental ERROR", () => {
    expect(isHiddenFromDiagnostics({ level: "ERROR", message: "you appear to be offline" })).toBe(
      true
    );
  });

  it("hides an environmental CRITICAL", () => {
    expect(
      isHiddenFromDiagnostics({ level: "CRITICAL", message: "you appear to be offline" })
    ).toBe(true);
  });

  it("never hides a WARNING, even one that would otherwise match the predicate", () => {
    expect(isHiddenFromDiagnostics({ level: "WARNING", message: "you appear to be offline" })).toBe(
      false
    );
  });

  it("does not hide a real ERROR", () => {
    expect(isHiddenFromDiagnostics({ level: "ERROR", message: "boom" })).toBe(false);
  });

  it("is false for a nullish entry", () => {
    expect(isHiddenFromDiagnostics(null)).toBe(false);
    expect(isHiddenFromDiagnostics(undefined)).toBe(false);
  });
});

describe("isCountableError", () => {
  it("counts an ERROR that is not environmental", () => {
    expect(isCountableError({ level: "ERROR", message: "boom" })).toBe(true);
  });

  it("counts CRITICAL, strictly worse than ERROR", () => {
    expect(isCountableError({ level: "CRITICAL", message: "boom" })).toBe(true);
  });

  it("does not count a WARNING", () => {
    expect(isCountableError({ level: "WARNING", message: "a fallback was taken" })).toBe(false);
  });

  it("does not count an environmental ERROR", () => {
    expect(isCountableError({ level: "ERROR", message: "you appear to be offline" })).toBe(false);
  });

  it("is false for a nullish entry", () => {
    expect(isCountableError(null)).toBe(false);
    expect(isCountableError(undefined)).toBe(false);
  });
});

describe("countNewErrors", () => {
  const since = Date.parse("2026-09-06T00:00:00.000Z");

  it("counts a real backend 500 as new", () => {
    const entries = [
      {
        level: "ERROR",
        timestamp: "2026-09-06T00:00:01.000Z",
        message: "GET /erudi/conversations -> 500 DATABASE_ERROR: boom",
      },
    ];
    expect(countNewErrors({ entries, since })).toBe(1);
  });

  it("excludes a record older than the cutoff", () => {
    const entries = [
      {
        level: "ERROR",
        timestamp: "2026-09-05T23:59:59.000Z",
        message: "boom",
      },
    ];
    expect(countNewErrors({ entries, since })).toBe(0);
  });

  it("excludes a record exactly at the cutoff", () => {
    const entries = [{ level: "ERROR", timestamp: "2026-09-06T00:00:00.000Z", message: "boom" }];
    expect(countNewErrors({ entries, since })).toBe(0);
  });

  it("excludes a WARNING even when new", () => {
    const entries = [
      { level: "WARNING", timestamp: "2026-09-06T00:00:05.000Z", message: "degraded" },
    ];
    expect(countNewErrors({ entries, since })).toBe(0);
  });

  it("excludes an environmental ERROR even when new", () => {
    const entries = [
      {
        level: "ERROR",
        timestamp: "2026-09-06T00:00:05.000Z",
        message: "you appear to be offline",
      },
    ];
    expect(countNewErrors({ entries, since })).toBe(0);
  });

  it("counts one entry once even though it repeated upstream (mergeRecentErrors' job, not this one's)", () => {
    // mergeRecentErrors already collapses repeats into a single entry with a
    // `count`; this function must not multiply by it.
    const entries = [
      {
        level: "ERROR",
        timestamp: "2026-09-06T00:00:05.000Z",
        message: "boom",
        count: 12,
      },
    ];
    expect(countNewErrors({ entries, since })).toBe(1);
  });

  it("ignores an entry with an unparsable timestamp rather than counting it", () => {
    const entries = [{ level: "ERROR", timestamp: "not-a-date", message: "boom" }];
    expect(countNewErrors({ entries, since })).toBe(0);
  });

  it("defaults to an empty list and a zero cutoff without throwing", () => {
    expect(countNewErrors()).toBe(0);
  });
});

describe("badgeLabel", () => {
  it("is empty at zero", () => {
    expect(badgeLabel(0)).toBe("");
  });

  it("is the exact number up to the cap", () => {
    expect(badgeLabel(1)).toBe("1");
    expect(badgeLabel(3)).toBe("3");
    expect(badgeLabel(BADGE_CAP)).toBe(String(BADGE_CAP));
  });

  it("reads 9+ beyond the cap", () => {
    expect(badgeLabel(BADGE_CAP + 1)).toBe("9+");
    expect(badgeLabel(12)).toBe("9+");
  });
});

function memoryStorage(initial = {}) {
  const store = new Map(Object.entries(initial));
  return {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
    clear: () => store.clear(),
  };
}

function installStorage(value) {
  Object.defineProperty(window, "localStorage", {
    value,
    configurable: true,
    writable: true,
  });
}

function throwingStorage(message = "storage blocked") {
  return {
    getItem: () => {
      throw new Error(message);
    },
    setItem: () => {
      throw new Error(message);
    },
    removeItem: () => {
      throw new Error(message);
    },
    clear: () => {
      throw new Error(message);
    },
  };
}

describe("diagnostics-visit tracking", () => {
  beforeEach(() => {
    installStorage(memoryStorage());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("has no visit recorded before one happens", () => {
    expect(getLastDiagnosticsVisit()).toBe(0);
  });

  it("persists the visit time and reads it back", () => {
    markDiagnosticsVisited(1000);
    expect(getLastDiagnosticsVisit()).toBe(1000);
    expect(window.localStorage.getItem(LAST_VISIT_STORAGE_KEY)).toBe("1000");
  });

  it("defaults to the current time when none is given", () => {
    const before = Date.now();
    const recorded = markDiagnosticsVisited();
    expect(recorded).toBeGreaterThanOrEqual(before);
    expect(getLastDiagnosticsVisit()).toBe(recorded);
  });

  it("notifies subscribers synchronously, for a badge to clear without a poll", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeDiagnosticsVisit(listener);
    markDiagnosticsVisited(2000);
    expect(listener).toHaveBeenCalledWith(2000);
    unsubscribe();
    markDiagnosticsVisited(3000);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("never throws when localStorage is unavailable, and still notifies subscribers", () => {
    installStorage(throwingStorage());
    const listener = vi.fn();
    const unsubscribe = subscribeDiagnosticsVisit(listener);
    expect(() => markDiagnosticsVisited(4000)).not.toThrow();
    expect(listener).toHaveBeenCalledWith(4000);
    unsubscribe();
  });

  it("reads 0 rather than throwing when localStorage.getItem throws", () => {
    installStorage(throwingStorage());
    expect(getLastDiagnosticsVisit()).toBe(0);
  });
});
