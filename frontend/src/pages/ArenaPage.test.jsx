// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// #136 C: attached images must reach the arena request payload (they were
// silently dropped: handleAsk ignored QuestionInput's images argument).
// #136 H: a running comparison must be stoppable — a Stop button aborts the
// in-flight fetches, panels keep their partial text and no error is surfaced.

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  apiClient: { get: vi.fn() },
  tracedFetch: tracedFetchMock,
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/GradientBox", () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock("../components/MarkdownRenderer", () => ({
  default: ({ content }) => <div>{content}</div>,
}));
vi.mock("../components/modals/CustomizePromptModal", () => ({ default: () => null }));
// #218: the panel header pushes slider edits live via onLiveChange (no Apply
// step). The mock exposes a per-panel "edit" control that fires onLiveChange
// with a distinctive settings pair so a test can prove the edited value
// reaches the outgoing request.
vi.mock("../components/HeaderBar", () => ({
  default: ({ currentModel, onLiveChange }) => (
    <div>
      <div>{`panel:${currentModel}`}</div>
      <button onClick={() => onLiveChange?.({ temperature: 0.49, topP: 0.5 })}>
        {`edit:${currentModel}`}
      </button>
    </div>
  ),
}));
// The composer is exercised through its contract: onSend(question, images,
// imagePaths) and the canAttachImages vision gate.
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend, canAttachImages }) => (
    <div>
      <div data-testid="can-attach">{String(canAttachImages)}</div>
      <button onClick={() => onSend("what is this?", ["data:image/png;base64,AAA"], [""])}>
        SEND_WITH_IMAGE
      </button>
      <button onClick={() => onSend("plain question", [], [])}>SEND_PLAIN</button>
    </div>
  ),
}));

import apiClient from "../services/api/client";
import ArenaPage from "./ArenaPage.jsx";

/** Minimal streaming Response stub: an already-finished text/plain body. */
const doneStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

const renderWithModels = async (models) => {
  apiClient.get.mockResolvedValue(models);
  render(<ArenaPage />);
  // Both initial panels are built once the models list resolves.
  await screen.findAllByText(`panel:${models[0].name}`);
};

beforeEach(() => {
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue(doneStream());
  apiClient.get.mockReset();
});
afterEach(() => {
  cleanup();
});

describe("ArenaPage images (#136 C)", () => {
  it("sends attached images in every panel's request payload", async () => {
    await renderWithModels([{ id: 1, name: "m1", supports_vision: true }]);

    fireEvent.click(screen.getByText("SEND_WITH_IMAGE"));

    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalledTimes(2));
    for (const [, options] of tracedFetchMock.mock.calls) {
      expect(JSON.parse(options.body).images).toEqual(["data:image/png;base64,AAA"]);
    }
    // The attachment stays visible in the user bubble of both panels.
    expect(screen.getAllByAltText("attachment 1")).toHaveLength(2);
  });

  it("enables image attach when at least one panel model supports vision", async () => {
    await renderWithModels([
      { id: 1, name: "m1", supports_vision: false },
      { id: 2, name: "m2", supports_vision: true },
    ]);
    expect(screen.getByTestId("can-attach").textContent).toBe("true");
  });

  it("disables image attach when every panel model is text-only", async () => {
    await renderWithModels([
      { id: 1, name: "m1", supports_vision: false },
      { id: 2, name: "m2", supports_vision: false },
    ]);
    expect(screen.getByTestId("can-attach").textContent).toBe("false");
  });
});

describe("ArenaPage live settings (#218)", () => {
  it("sends a panel's edited settings without an Apply step, and does not leak across panels", async () => {
    // Two distinct models so each panel's request has a distinguishable URL
    // (/arena/<llmId>/query).
    await renderWithModels([
      { id: 1, name: "m1", supports_vision: true },
      { id: 2, name: "m2", supports_vision: true },
    ]);

    // Edit only the first panel (model m1 -> llmId 1). No Apply is clicked.
    fireEvent.click(screen.getByText("edit:m1"));

    fireEvent.click(screen.getByText("SEND_PLAIN"));

    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalledTimes(2));

    const bodyByUrl = Object.fromEntries(
      tracedFetchMock.mock.calls.map(([url, options]) => [url, JSON.parse(options.body)])
    );
    const bodyFor = (llmId) =>
      Object.entries(bodyByUrl).find(([url]) => url.includes(`/arena/${llmId}/query`))[1];

    // The edited panel's request carries the live-edited values...
    const edited = bodyFor(1);
    expect(edited.temperature).toBe(0.49);
    expect(edited.top_p).toBe(0.5);
    // The output budget is not a panel setting any more: nothing sends it.
    expect(edited.max_new_tokens).toBeUndefined();

    // ...while the untouched panel keeps its model's defaults (no cross-panel
    // leak): a model without hints seeds from the backend fallback (#388).
    const untouched = bodyFor(2);
    expect(untouched.temperature).toBe(0.2);
    expect(untouched.top_p).toBe(0.95);
    expect(untouched.max_new_tokens).toBeUndefined();
  });
});

describe("ArenaPage stop generation (#136 H)", () => {
  it("aborts in-flight requests, clears loading and surfaces no error", async () => {
    // Never-resolving fetch that rejects with AbortError on signal abort.
    const seenSignals = [];
    tracedFetchMock.mockImplementation(
      (url, options) =>
        new Promise((resolve, reject) => {
          seenSignals.push(options.signal);
          options.signal?.addEventListener("abort", () =>
            reject(Object.assign(new Error("aborted"), { name: "AbortError" }))
          );
        })
    );
    await renderWithModels([{ id: 1, name: "m1", supports_vision: true }]);

    fireEvent.click(screen.getByText("SEND_PLAIN"));

    // While generating, a Stop control is offered and both requests are abortable.
    const stopButton = await screen.findByLabelText("Stop generation");
    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalledTimes(2));
    expect(seenSignals.every((s) => s instanceof AbortSignal)).toBe(true);

    fireEvent.click(stopButton);

    // Loading clears (the Stop button goes away with it)…
    await waitFor(() => expect(screen.queryByLabelText("Stop generation")).toBeNull());
    // …and after the flush interval had time to drain, no error was surfaced.
    await new Promise((r) => setTimeout(r, 150));
    expect(screen.queryByText(/\[Erreur\]/)).toBeNull();
  });
});
