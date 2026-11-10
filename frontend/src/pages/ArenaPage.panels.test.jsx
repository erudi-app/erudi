// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// Panel lifecycle and per-panel failure isolation of the arena: a failing
// model shows an error bubble in ITS panel only while the other panel keeps
// its streamed answer; an unknown model reports "[Model not found]" without
// firing a request; panels can be added up to 4 and deleted down to 1; a
// panel's custom prompt reaches only that panel's request payload.

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
// Header mock exposing the panel identity plus the model-change and
// customize-prompt hooks this suite drives.
vi.mock("../components/HeaderBar", () => ({
  default: ({ currentModel, onModelChange, onCustomizePrompt }) => (
    <div>
      <div>{`panel:${currentModel}`}</div>
      <button onClick={() => onModelChange("ghost-model")}>{`ghost:${currentModel}`}</button>
      <button onClick={() => onCustomizePrompt()}>{`prompt:${currentModel}`}</button>
    </div>
  ),
}));
vi.mock("../components/modals/CustomizePromptModal", () => ({
  default: ({ isOpen, onSave, onClose }) =>
    isOpen ? (
      <div>
        <button
          onClick={() => {
            onSave("Speak only Latin");
            onClose();
          }}
        >
          save-prompt
        </button>
      </div>
    ) : null,
}));
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => (
    <div>
      <button onClick={() => onSend("plain question", [], [])}>SEND_PLAIN</button>
      <button onClick={() => onSend("   ", [], [])}>SEND_EMPTY</button>
    </div>
  ),
}));

import apiClient from "../services/api/client";
import ArenaPage from "./ArenaPage.jsx";

const encoder = new TextEncoder();

/** Streaming Response stub: one NDJSON answer event per given text. */
const streamOf = (...chunks) => {
  const queue = chunks.map((text) => ({
    done: false,
    value: encoder.encode(`${JSON.stringify({ t: "answer", text })}\n`),
  }));
  queue.push({ done: true, value: undefined });
  return {
    ok: true,
    body: { getReader: () => ({ read: async () => queue.shift() }) },
  };
};

const MODELS = [
  { id: 1, name: "m1", supports_vision: false },
  { id: 2, name: "m2", supports_vision: false },
];

const renderArena = async (models = MODELS) => {
  apiClient.get.mockResolvedValue(models);
  render(<ArenaPage />);
  await screen.findAllByText(`panel:${models[0].name}`);
};

beforeEach(() => {
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue(streamOf("ok"));
  apiClient.get.mockReset();
});
afterEach(() => {
  cleanup();
});

describe("ArenaPage per-panel failure isolation", () => {
  it("shows an error bubble only in the failing panel; the other keeps its answer", async () => {
    tracedFetchMock.mockImplementation(async (url) =>
      String(url).includes("/arena/2/") ? { ok: false, status: 500 } : streamOf("Paris")
    );
    await renderArena();

    fireEvent.click(screen.getByText("SEND_PLAIN"));

    // The failing panel surfaces the error marker...
    expect(await screen.findByText(/\[Error\]/)).toBeTruthy();
    // ...while the healthy panel still shows its streamed answer.
    expect(await screen.findByText("Paris")).toBeTruthy();
    expect(screen.getAllByText(/\[Error\]/)).toHaveLength(1);
  });

  it("reports [Model not found] without any request for an unknown panel model", async () => {
    await renderArena();

    // Point the first panel at a model that no longer exists locally.
    fireEvent.click(screen.getByText("ghost:m1"));
    await screen.findByText("panel:ghost-model");

    fireEvent.click(screen.getByText("SEND_PLAIN"));

    expect(await screen.findByText(/\[Model not found\]/)).toBeTruthy();
    // Only the healthy panel fired a request.
    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalledTimes(1));
    expect(String(tracedFetchMock.mock.calls[0][0])).toContain("/arena/2/");
  });

  it("ignores an empty ask entirely", async () => {
    await renderArena();

    fireEvent.click(screen.getByText("SEND_EMPTY"));

    await new Promise((r) => setTimeout(r, 50));
    expect(tracedFetchMock).not.toHaveBeenCalled();
  });
});

describe("ArenaPage panel management", () => {
  it("adds panels up to the maximum of 4, then disables the add button", async () => {
    await renderArena();
    expect(screen.getAllByText(/^panel:/)).toHaveLength(2);

    const addButton = screen.getByTitle("Add chat panel");
    fireEvent.click(addButton);
    await waitFor(() => expect(screen.getAllByText(/^panel:/)).toHaveLength(3));
    fireEvent.click(addButton);
    await waitFor(() => expect(screen.getAllByText(/^panel:/)).toHaveLength(4));

    const maxedButton = screen.getByTitle("Maximum 4 panels");
    expect(maxedButton.disabled).toBe(true);
    fireEvent.click(maxedButton);
    expect(screen.getAllByText(/^panel:/)).toHaveLength(4);
  });

  it("deletes a panel but never the last one", async () => {
    await renderArena([{ id: 1, name: "m1", supports_vision: false }]);
    expect(screen.getAllByText(/^panel:/)).toHaveLength(2);

    const [firstDelete] = screen.getAllByTitle("Delete this panel");
    fireEvent.click(firstDelete);
    await waitFor(() => expect(screen.getAllByText(/^panel:/)).toHaveLength(1), {
      timeout: 2000,
    });

    // The survivor cannot be deleted.
    const lastDelete = screen.getByTitle("Cannot delete the last panel");
    expect(lastDelete.disabled).toBe(true);
    fireEvent.click(lastDelete);
    await new Promise((r) => setTimeout(r, 350));
    expect(screen.getAllByText(/^panel:/)).toHaveLength(1);
  });
});

describe("ArenaPage per-panel custom prompt", () => {
  it("sends the saved prompt only in the edited panel's payload", async () => {
    await renderArena();

    fireEvent.click(screen.getByText("prompt:m1"));
    fireEvent.click(await screen.findByText("save-prompt"));
    await waitFor(() => expect(screen.queryByText("save-prompt")).toBeNull());

    fireEvent.click(screen.getByText("SEND_PLAIN"));
    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalledTimes(2));

    const bodyFor = (llmId) =>
      JSON.parse(
        tracedFetchMock.mock.calls.find(([url]) => String(url).includes(`/arena/${llmId}/`))[1].body
      );
    expect(bodyFor(1).custom_prompt).toBe("Speak only Latin");
    expect(bodyFor(2).custom_prompt).toBe("");
  });
});
