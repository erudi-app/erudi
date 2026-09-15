// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// Arena panels are model-scoped (#388): a panel seeds its sampling from its
// model's `sampling_defaults`, a model change resets the panel to the new
// model's defaults, a new panel takes its model's defaults, and the request
// payload carries what the panel shows. The unvalidated 1.0 / 0.95 / 512
// constants are gone: a model without hints gets the backend fallback.

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
// Header mock surfacing the sampling the panel hands it, plus a model switch.
vi.mock("../components/HeaderBar", () => ({
  default: ({ currentModel, initialTemperature, initialTopP, onModelChange, onLiveChange }) => (
    <div>
      <div data-testid="panel">{`${currentModel}|${initialTemperature}|${initialTopP}`}</div>
      <button onClick={() => onModelChange("plain")}>{`switch:${currentModel}`}</button>
      <button onClick={() => onLiveChange({ temperature: 1.3, topP: initialTopP })}>
        {`touch:${currentModel}`}
      </button>
    </div>
  ),
}));
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => <button onClick={() => onSend("q", [], [])}>SEND</button>,
}));

import apiClient from "../services/api/client";
import ArenaPage from "./ArenaPage.jsx";

const encoder = new TextEncoder();
const streamOf = (...chunks) => {
  const queue = chunks.map((text) => ({ done: false, value: encoder.encode(text) }));
  queue.push({ done: true, value: undefined });
  return { ok: true, body: { getReader: () => ({ read: async () => queue.shift() }) } };
};

const MODELS = [
  {
    id: 1,
    name: "qwen3",
    supports_vision: false,
    sampling_defaults: {
      temperature: 0.6,
      top_p: 0.95,
      max_tokens: 1024,
      max_tokens_cap: 8192,
      top_k: 20,
      source: "base_generation_config",
    },
  },
  { id: 2, name: "plain", supports_vision: false },
];

const panels = () => screen.getAllByTestId("panel").map((n) => n.textContent);

const renderArena = async () => {
  apiClient.get.mockResolvedValue(MODELS);
  render(<ArenaPage />);
  await screen.findAllByTestId("panel");
};

beforeEach(() => {
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue(streamOf("ok"));
  apiClient.get.mockReset();
});
afterEach(() => {
  cleanup();
});

describe("ArenaPage per-model sampling defaults (#388)", () => {
  it("seeds each panel from its model; a model without hints gets the fallback", async () => {
    await renderArena();
    expect(panels()).toEqual(["qwen3|0.6|0.95", "plain|0.2|0.95"]);
  });

  it("resets a panel's sampling to the new model's defaults on switch, even when touched", async () => {
    await renderArena();

    fireEvent.click(screen.getByText("touch:qwen3"));
    await waitFor(() => expect(panels()[0]).toBe("qwen3|1.3|0.95"));

    fireEvent.click(screen.getByText("switch:qwen3"));
    await waitFor(() => expect(panels()[0]).toBe("plain|0.2|0.95"));
  });

  it("gives a new panel its model's defaults", async () => {
    await renderArena();

    fireEvent.click(screen.getByTitle("Add chat panel"));
    await waitFor(() => expect(panels()).toHaveLength(3));
    // The third panel cycles back onto models[0].
    expect(panels()[2]).toBe("qwen3|0.6|0.95");
  });

  it("sends the model's defaults in the arena payload", async () => {
    await renderArena();

    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalledTimes(2));

    const bodyFor = (llmId) =>
      JSON.parse(
        tracedFetchMock.mock.calls.find(([url]) => String(url).includes(`/arena/${llmId}/`))[1].body
      );
    expect([bodyFor(1).temperature, bodyFor(1).top_p]).toEqual([0.6, 0.95]);
    expect([bodyFor(2).temperature, bodyFor(2).top_p]).toEqual([0.2, 0.95]);
    // No output budget travels with the turn: the backend derives it from the
    // context window the model runs in.
    expect(bodyFor(1).max_new_tokens).toBeUndefined();
  });
});
