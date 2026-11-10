// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act } from "@testing-library/react";

// The KB-assistant creation context was untested. These drive its lifecycle
// (confirm -> POST create -> 2s status poll) and assert the state handoff
// (isStarting -> isCreating), the completed/failed/error callback routing, and
// that closeModal tears down the poll interval so a stale timer can't fire a
// late callback after the user dismissed the flow.
//
// tracedFetch is mocked; fake timers drive the setInterval poll deterministically.

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  apiClient: { get: vi.fn() },
  tracedFetch: tracedFetchMock,
}));
vi.mock("../components/Spinner", () => ({ default: () => null }));

import { KnowledgeBaseProvider, useKnowledgeBase } from "./KnowledgeBaseContext";

let createResponder;
let statusResponder;

const onComplete = vi.fn();
const onError = vi.fn();
let currentTask;

function Consumer() {
  const ctx = useKnowledgeBase();
  return (
    <>
      <button onClick={() => ctx.open(currentTask, { onComplete, onError })}>OPEN</button>
      <button onClick={() => ctx.closeModal()}>CLOSE</button>
      <span data-testid="creating">{String(ctx.isCreating)}</span>
      <span data-testid="starting">{String(ctx.isStarting)}</span>
    </>
  );
}

const renderProvider = () =>
  render(
    <KnowledgeBaseProvider>
      <Consumer />
    </KnowledgeBaseProvider>
  );

const val = (id) => screen.getByTestId(id).textContent;

const statusCalls = () =>
  tracedFetchMock.mock.calls.filter(
    ([u]) => String(u).includes("/knowledge_base/") && String(u).endsWith("/status")
  );

// open -> confirm the inline modal -> let the create POST resolve and fire one poll.
const startAndPollOnce = async () => {
  await act(async () => {
    fireEvent.click(screen.getByText("OPEN"));
  });
  await act(async () => {
    fireEvent.click(screen.getByText("Create Assistant"));
    await vi.advanceTimersByTimeAsync(2000);
  });
};

beforeEach(() => {
  vi.useFakeTimers();
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});

  onComplete.mockReset();
  onError.mockReset();
  currentTask = {
    modelName: "My KB",
    paths: ["/a.pdf", "/b.pdf"],
    selectedModel: 1,
    description: "docs",
  };

  createResponder = () => ({ ok: true, json: async () => ({ model_id: 7 }) });
  statusResponder = () => ({ ok: true, json: async () => ({ status: "completed" }) });

  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    if (method === "POST" && u.endsWith("/knowledge_base/create")) {
      return createResponder();
    }
    if (u.includes("/knowledge_base/") && u.endsWith("/status")) {
      return statusResponder();
    }
    return { ok: true, json: async () => ({}) };
  });
});

afterEach(() => {
  cleanup();
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("KnowledgeBaseContext creation success", () => {
  it("POSTs the create request with the task payload then fires onComplete", async () => {
    renderProvider();
    await startAndPollOnce();

    const post = tracedFetchMock.mock.calls.find(([, o]) => o?.method === "POST");
    expect(String(post[0])).toMatch(/\/knowledge_base\/create$/);
    const body = JSON.parse(post[1].body);
    expect(body).toMatchObject({
      paths: ["/a.pdf", "/b.pdf"],
      selectedModel: 1,
      modelName: "My KB",
      description: "docs",
    });

    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
    expect(val("creating")).toBe("false");
  });

  it("hands the files that failed to onComplete, empty when none did", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "completed", failed_files: ["maintenance.docx", "b.pdf"] }),
    });
    renderProvider();
    await startAndPollOnce();
    expect(onComplete).toHaveBeenCalledWith({ failedFiles: ["maintenance.docx", "b.pdf"] });

    cleanup();
    onComplete.mockReset();
    statusResponder = () => ({ ok: true, json: async () => ({ status: "completed" }) });
    renderProvider();
    await startAndPollOnce();
    expect(onComplete).toHaveBeenCalledWith({ failedFiles: [] });
  });

  it("hands the failed files to onError when the whole job failed", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({
        status: "failed",
        error_message: "No searchable content could be indexed",
        failed_files: ["maintenance.docx"],
      }),
    });
    renderProvider();
    await startAndPollOnce();

    expect(onError).toHaveBeenCalledWith("No searchable content could be indexed", {
      failedFiles: ["maintenance.docx"],
    });
    expect(onComplete).not.toHaveBeenCalled();
  });

  it("hands off from isStarting to isCreating once the create call returns", async () => {
    statusResponder = () => ({ ok: true, json: async () => ({ status: "running" }) });
    renderProvider();

    await startAndPollOnce();

    // The button spinner (isStarting) is cleared and the bottom-left creating
    // state is on while the poll keeps reporting 'running'.
    expect(val("starting")).toBe("false");
    expect(val("creating")).toBe("true");
    expect(onComplete).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });
});

describe("KnowledgeBaseContext confirmation wording (#317)", () => {
  it("asks to create when the task is a new assistant", async () => {
    renderProvider();
    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });

    expect(screen.getByText(/Are you sure you want to create/)).toBeTruthy();
    expect(screen.getByText("Create Assistant")).toBeTruthy();
    expect(screen.queryByText(/knowledge base with/)).toBeNull();
  });

  it("asks to update when the task targets an existing assistant", async () => {
    currentTask = { ...currentTask, isUpdate: true };
    renderProvider();
    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });

    // "Update <name>'s knowledge base with 2 file(s)?" — never "create".
    expect(screen.getByText(/knowledge base with 2 file\(s\)\?/)).toBeTruthy();
    expect(screen.getByText("My KB")).toBeTruthy();
    expect(screen.queryByText(/Are you sure you want to create/)).toBeNull();
    expect(screen.queryByText("Create Assistant")).toBeNull();

    // The confirm button still drives the same POST /create flow.
    await act(async () => {
      fireEvent.click(screen.getByText("Update Knowledge Base"));
      await vi.advanceTimersByTimeAsync(2000);
    });
    const post = tracedFetchMock.mock.calls.find(([, o]) => o?.method === "POST");
    expect(String(post[0])).toMatch(/\/knowledge_base\/create$/);
  });
});

describe("KnowledgeBaseContext failure routing", () => {
  it("routes a failed status to onError with the backend message", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "failed", error_message: "ingestion crashed" }),
    });
    renderProvider();

    await startAndPollOnce();

    expect(onError).toHaveBeenCalledWith("ingestion crashed", { failedFiles: [] });
    expect(onComplete).not.toHaveBeenCalled();
    expect(val("creating")).toBe("false");
  });

  it("falls back to a default message when a failure carries no error_message", async () => {
    statusResponder = () => ({ ok: true, json: async () => ({ status: "failed" }) });
    renderProvider();

    await startAndPollOnce();

    expect(onError).toHaveBeenCalledWith("Assistant creation failed unexpectedly", {
      failedFiles: [],
    });
  });

  it("surfaces a status poll error through onError and stops creating", async () => {
    statusResponder = () => {
      throw new TypeError("Failed to fetch");
    };
    renderProvider();

    await startAndPollOnce();

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0]).toContain("An error occurred during assistant creation");
    expect(val("creating")).toBe("false");
  });

  it("reports a failed create START through onError and never begins polling", async () => {
    createResponder = () => ({
      ok: false,
      status: 400,
      json: async () => ({ detail: "bad paths" }),
    });
    renderProvider();

    await startAndPollOnce();

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0]).toContain("Failed to start assistant creation (400)");
    expect(val("creating")).toBe("false");
    expect(statusCalls()).toHaveLength(0);
  });
});

describe("KnowledgeBaseContext closeModal", () => {
  it("tears down the poll interval so no late callback fires after dismiss", async () => {
    statusResponder = () => ({ ok: true, json: async () => ({ status: "running" }) });
    renderProvider();
    await startAndPollOnce();

    const pollsBeforeClose = statusCalls().length;
    expect(pollsBeforeClose).toBeGreaterThanOrEqual(1);
    expect(val("creating")).toBe("true");

    await act(async () => {
      fireEvent.click(screen.getByText("CLOSE"));
    });
    // Advance well past several would-be poll ticks.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(8000);
    });

    expect(statusCalls().length).toBe(pollsBeforeClose); // interval was cleared
    expect(val("creating")).toBe("false");
    expect(onComplete).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });
});
