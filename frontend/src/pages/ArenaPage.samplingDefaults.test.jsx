// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// Arena panels follow their model's defaults (#609, #627): a panel shows the
// values its model starts from (`GET /llms/{id}/conversation_defaults`: saved
// with /config global, else recommended), re-reads them when its model
// changes, and leaves to the backend every value the user did not move: an
// untouched slider sends null, a moved one sends its value. When the defaults
// cannot be read, the sliders show the model's recommended sampling.

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
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
  const queue = chunks.map((text) => ({
    done: false,
    value: encoder.encode(`${JSON.stringify({ t: "answer", text })}\n`),
  }));
  queue.push({ done: true, value: undefined });
  return { ok: true, body: { getReader: () => ({ read: async () => queue.shift() }) } };
};

const MODELS = [
  {
    id: 1,
    name: "qwen3",
    supports_vision: false,
    sampling_defaults: { temperature: 0.6, top_p: 0.95, source: "base_generation_config" },
  },
  { id: 2, name: "plain", supports_vision: false },
];

const defaults = (temperature, topP) => ({
  reasoning_effort: { value: "medium", source: "settings" },
  temperature: { value: temperature, source: "saved" },
  top_p: { value: topP, source: "saved" },
  compaction_threshold: { value: 80, source: "settings" },
  web_search_enabled: { value: false, source: "settings" },
  custom_prompt: { value: "", source: "fallback" },
});

let defaultsFor;

const panels = () => screen.getAllByTestId("panel").map((n) => n.textContent);

const bodyFor = (llmId) =>
  JSON.parse(
    tracedFetchMock.mock.calls.find(([url]) => String(url).includes(`/arena/${llmId}/`))[1].body
  );

const renderArena = async () => {
  render(<ArenaPage />);
  await screen.findAllByTestId("panel");
};

beforeEach(() => {
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue(streamOf("ok"));
  defaultsFor = { 1: async () => defaults(0.9, 0.7), 2: async () => defaults(0.4, 0.5) };
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async (path) => {
    if (path === "/llms/local") return MODELS;
    const match = path.match(/^\/llms\/(\d+)\/conversation_defaults$/);
    if (match) return defaultsFor[match[1]]();
    throw new Error(`unexpected ${path}`);
  });
});
afterEach(() => {
  cleanup();
});

describe("ArenaPage per-model defaults", () => {
  it("shows each panel's model defaults once they are read", async () => {
    await renderArena();
    await waitFor(() => expect(panels()).toEqual(["qwen3|0.9|0.7", "plain|0.4|0.5"]));
    expect(apiClient.get).toHaveBeenCalledWith("/llms/1/conversation_defaults");
    expect(apiClient.get).toHaveBeenCalledWith("/llms/2/conversation_defaults");
  });

  it("sends null for every value of an untouched panel", async () => {
    await renderArena();
    await waitFor(() => expect(panels()[0]).toBe("qwen3|0.9|0.7"));
    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalledTimes(2));
    for (const llmId of [1, 2]) {
      const body = bodyFor(llmId);
      expect(body.temperature).toBeNull();
      expect(body.top_p).toBeNull();
      expect(body.reasoning_effort ?? null).toBeNull();
      expect(body.max_new_tokens).toBeUndefined();
    }
  });

  it("sends the value a user moved, and only in that panel", async () => {
    await renderArena();
    await waitFor(() => expect(panels()[0]).toBe("qwen3|0.9|0.7"));
    fireEvent.click(screen.getByText("touch:qwen3"));
    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalledTimes(2));
    expect(bodyFor(1).temperature).toBe(1.3);
    expect(bodyFor(1).top_p).toBeNull();
    expect(bodyFor(2).temperature).toBeNull();
  });

  it("re-reads the defaults on a model change and forgets what was moved", async () => {
    await renderArena();
    await waitFor(() => expect(panels()[0]).toBe("qwen3|0.9|0.7"));
    fireEvent.click(screen.getByText("touch:qwen3"));
    await waitFor(() => expect(panels()[0]).toBe("qwen3|1.3|0.7"));

    fireEvent.click(screen.getByText("switch:qwen3"));
    await waitFor(() => expect(panels()[0]).toBe("plain|0.4|0.5"));

    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalledTimes(2));
    for (const [, options] of tracedFetchMock.mock.calls) {
      expect(JSON.parse(options.body).temperature).toBeNull();
    }
  });

  it("ignores a late answer for the panel's previous model", async () => {
    let releaseQwen;
    defaultsFor[1] = () =>
      new Promise((resolve) => {
        releaseQwen = () => resolve(defaults(1.9, 0.1));
      });
    await renderArena();
    fireEvent.click(screen.getByText("switch:qwen3"));
    await waitFor(() => expect(panels()[0]).toBe("plain|0.4|0.5"));
    await act(async () => releaseQwen());
    expect(panels()[0]).toBe("plain|0.4|0.5");
  });

  it("shows the model's recommended values when its defaults cannot be read, and still sends null", async () => {
    defaultsFor[1] = async () => {
      throw new Error("offline");
    };
    await renderArena();
    await waitFor(() => expect(panels()[1]).toBe("plain|0.4|0.5"));
    expect(panels()[0]).toBe("qwen3|0.6|0.95");

    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalledTimes(2));
    expect(bodyFor(1).temperature).toBeNull();
    expect(bodyFor(1).top_p).toBeNull();
  });

  it("gives a new panel its model's defaults", async () => {
    await renderArena();
    fireEvent.click(screen.getByTitle("Add chat panel"));
    await waitFor(() => expect(panels()).toHaveLength(3));
    await waitFor(() => expect(panels()[2]).toBe("qwen3|0.9|0.7"));
  });
});
