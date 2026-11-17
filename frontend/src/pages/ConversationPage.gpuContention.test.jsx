// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// The amber `gpu_contention` notice: another app is taking the graphics card
// from the model (Windows, the CUDA engine's eviction gauge).
//  - a `memory_warning` event with `reason: "gpu_contention"` shows its copy,
//    which quotes no size;
//  - while that notice shows, and only then, the page polls
//    `GET /hardware/gpu_contention/status` about every 10 s (a setTimeout
//    chain, paused while a turn runs) and drops the notice once `active` is
//    false; a failed poll, or `active: null` (no reading), keeps it;
//  - the poll only ever clears a gpu_contention notice: a warning of another
//    reason that replaced it stays;
//  - the notice carries "Free the graphics card", which unloads the
//    conversation's model (`POST /llms/loaded/unload`), is disabled while an
//    answer is written, says why on a 409, and hides the notice once the
//    model is unloaded.

const { tracedFetchMock, navigateMock, locationMock, paramsMock, statusMock, postMock } =
  vi.hoisted(() => ({
    tracedFetchMock: vi.fn(),
    navigateMock: vi.fn(),
    locationMock: { pathname: "/conversation/7", state: null },
    paramsMock: { id: "7" },
    statusMock: vi.fn(),
    postMock: vi.fn(),
  }));

const STATUS_PATH = "/hardware/gpu_contention/status";

vi.mock("../services/api/client", () => {
  const get = vi.fn(async (path) => (path === STATUS_PATH ? statusMock() : []));
  const client = { get, post: postMock };
  return { default: client, apiClient: client, tracedFetch: tracedFetchMock };
});

vi.mock("react-router-dom", () => ({
  useParams: () => ({ id: paramsMock.id }),
  useNavigate: () => navigateMock,
  useLocation: () => locationMock,
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/ConversationHeader", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/TraceStrip", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({
  default: ({ content }) => <div data-testid="answer">{content}</div>,
}));
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => <button onClick={() => onSend("hi", [], [])}>SEND</button>,
}));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";

const CONTENTION =
  "Another app is taking your graphics card's memory: answers are slower until it frees " +
  "some. Freeing the card unloads the model; your next message reloads it, with only what " +
  "fits on the card then. Nothing is lost.";
const FREE = "Free the graphics card";
const BUSY = "The model is busy; try again in a moment.";
const CHANGED = "Another model is loaded now.";
const POLL_MS = 10_000;

const conversationDetail = {
  id: 7,
  llm_id: 1,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
};

/** A streaming Response whose body reader is fed chunks on demand. */
function makeControlledStream() {
  const enc = new TextEncoder();
  const queue = [];
  let waiting = null;
  let ended = false;
  const push = (str) => {
    const chunk = { done: false, value: enc.encode(str) };
    if (waiting) {
      const resolve = waiting;
      waiting = null;
      resolve(chunk);
    } else {
      queue.push(chunk);
    }
  };
  const end = () => {
    ended = true;
    if (waiting) {
      const resolve = waiting;
      waiting = null;
      resolve({ done: true, value: undefined });
    }
  };
  const read = () =>
    new Promise((resolve) => {
      if (queue.length) {
        resolve(queue.shift());
      } else if (ended) {
        resolve({ done: true, value: undefined });
      } else {
        waiting = resolve;
      }
    });
  return { response: { ok: true, body: { getReader: () => ({ read }) } }, push, end };
}

const doneStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

const makeRoute = (queryStreams) => {
  const pending = [...queryStreams];
  return async (url) => {
    const u = String(url);
    if (u.includes("/query")) return pending.shift().response;
    if (u.includes("generate_title")) return doneStream();
    if (/\/conversations\/\d+$/.test(u)) {
      return { ok: true, json: async () => conversationDetail };
    }
    return { ok: true, json: async () => [] };
  };
};

const renderAndSettle = async () => {
  const view = render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
  await screen.findByText("SEND");
  return view;
};

const runTurn = async (stream, lines) => {
  await act(async () => {
    for (const line of lines) {
      stream.push(line);
    }
    stream.end();
  });
};

const contendedTurn = (stream) =>
  runTurn(stream, [
    '{"t":"answer","text":"Slow."}\n',
    '{"t":"memory_warning","reason":"gpu_contention"}\n',
    '{"t":"done"}\n',
  ]);

const statusCalls = () => apiClient.get.mock.calls.filter(([path]) => path === STATUS_PATH);

const elapse = async (ms) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
};

const httpError = (status, type) =>
  Object.assign(new Error(`HTTP ${status}`), {
    status,
    data: type ? { error: { type } } : {},
  });

beforeEach(() => {
  vi.clearAllMocks();
  postMock.mockReset();
  paramsMock.id = "7";
  statusMock.mockImplementation(async () => ({ active: true }));
  Element.prototype.scrollTo = () => {};
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("ConversationPage gpu contention notice", () => {
  it("renders its copy, with no size to quote", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await contendedTurn(turn);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain(CONTENTION);
  });

  it("polls about every 10 s while it shows, and hides once the contention ends", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await contendedTurn(turn);
    await screen.findByRole("alert");
    expect(statusCalls()).toHaveLength(0);

    await elapse(POLL_MS);
    expect(statusCalls()).toHaveLength(1);
    expect(screen.getByRole("alert").textContent).toContain(CONTENTION);

    await elapse(POLL_MS);
    expect(statusCalls()).toHaveLength(2);

    statusMock.mockImplementation(async () => ({ active: false }));
    await elapse(POLL_MS);
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    expect(statusCalls()).toHaveLength(3);

    // Gone: nothing polls any more.
    await elapse(5 * POLL_MS);
    expect(statusCalls()).toHaveLength(3);
  });

  it("keeps the notice and keeps polling when a poll fails", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await contendedTurn(turn);
    await screen.findByRole("alert");

    statusMock.mockImplementation(async () => {
      throw new Error("backend busy");
    });
    await elapse(POLL_MS);
    await elapse(POLL_MS);
    expect(statusCalls()).toHaveLength(2);
    expect(screen.getByRole("alert").textContent).toContain(CONTENTION);
  });

  it("keeps the notice and keeps polling while the status is unknown (active: null)", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await contendedTurn(turn);
    await screen.findByRole("alert");

    // No reading (a model the engine just restarted, not read yet) is not
    // the end of the contention.
    statusMock.mockImplementation(async () => ({ active: null }));
    await elapse(POLL_MS);
    await elapse(POLL_MS);
    expect(statusCalls()).toHaveLength(2);
    expect(screen.getByRole("alert").textContent).toContain(CONTENTION);

    statusMock.mockImplementation(async () => ({ active: false }));
    await elapse(POLL_MS);
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
  });

  it("does not poll while a turn runs", async () => {
    const first = makeControlledStream();
    const second = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([first, second]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await contendedTurn(first);
    await screen.findByRole("alert");

    fireEvent.click(screen.getByText("SEND"));
    await act(async () => {
      second.push('{"t":"answer","text":"Still slow"}\n');
    });
    await elapse(3 * POLL_MS);
    expect(statusCalls()).toHaveLength(0);

    await act(async () => {
      second.push('{"t":"memory_warning","reason":"gpu_contention"}\n');
      second.push('{"t":"done"}\n');
      second.end();
    });
    await elapse(POLL_MS);
    expect(statusCalls()).toHaveLength(1);
  });

  it("never polls for a notice of another reason", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [
      '{"t":"answer","text":"Long."}\n',
      '{"t":"memory_warning","reason":"bound","footprint_bytes":1048576}\n',
      '{"t":"done"}\n',
    ]);
    await screen.findByRole("alert");

    await elapse(5 * POLL_MS);
    expect(statusCalls()).toHaveLength(0);
  });

  it("never clears a bound warning that replaced the contention notice", async () => {
    const first = makeControlledStream();
    const second = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([first, second]));
    statusMock.mockImplementation(async () => ({ active: false }));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await contendedTurn(first);
    await screen.findByRole("alert");

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(second, [
      '{"t":"answer","text":"Long."}\n',
      '{"t":"memory_warning","reason":"bound","footprint_bytes":1048576}\n',
      '{"t":"done"}\n',
    ]);
    await elapse(5 * POLL_MS);

    const alert = screen.getByRole("alert");
    expect(alert.textContent).toContain("Your computer's memory limits this conversation");
  });

  it("stops polling when the user switches conversation", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    const view = await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await contendedTurn(turn);
    await screen.findByRole("alert");

    paramsMock.id = "8";
    view.rerender(<ConversationPage />);
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());

    await elapse(5 * POLL_MS);
    expect(statusCalls()).toHaveLength(0);
  });

  it("shows from the turn's start, while the answer is written, and stays at its end", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await act(async () => {
      turn.push('{"t":"memory_warning","reason":"gpu_contention"}\n');
    });
    expect((await screen.findByRole("alert")).textContent).toContain(CONTENTION);

    // The backend does not say it again at the end of the same turn.
    await runTurn(turn, ['{"t":"answer","text":"Slow."}\n', '{"t":"done"}\n']);
    expect(screen.getByRole("alert").textContent).toContain(CONTENTION);
  });

  it("offers to free the graphics card, disabled while the answer is written", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await act(async () => {
      turn.push('{"t":"memory_warning","reason":"gpu_contention"}\n');
    });
    const action = await screen.findByRole("button", { name: FREE });
    expect(action.disabled).toBe(true);
    fireEvent.click(action);
    expect(postMock).not.toHaveBeenCalled();

    await runTurn(turn, ['{"t":"answer","text":"Slow."}\n', '{"t":"done"}\n']);
    await waitFor(() => expect(screen.getByRole("button", { name: FREE }).disabled).toBe(false));
  });

  it("unloads the conversation's model and hides the notice", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    postMock.mockResolvedValue(undefined);
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await contendedTurn(turn);
    fireEvent.click(await screen.findByRole("button", { name: FREE }));

    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    expect(postMock).toHaveBeenCalledTimes(1);
    expect(postMock).toHaveBeenCalledWith("/llms/loaded/unload", { llm_id: 1 });
  });

  it.each([
    ["busy", httpError(409), BUSY],
    ["another model loaded", httpError(409, "LOADED_MODEL_CHANGED"), CHANGED],
  ])("says why on a 409 (%s) and keeps the notice", async (_label, error, sentence) => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    postMock.mockRejectedValue(error);
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await contendedTurn(turn);
    fireEvent.click(await screen.findByRole("button", { name: FREE }));

    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(sentence));
    expect(screen.getByRole("alert").textContent).toContain(CONTENTION);
    expect(screen.getByRole("button", { name: FREE }).disabled).toBe(false);
  });

  it("offers no action on a warning of another reason", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [
      '{"t":"answer","text":"Long."}\n',
      '{"t":"memory_warning","reason":"bound","footprint_bytes":1048576}\n',
      '{"t":"done"}\n',
    ]);
    await screen.findByRole("alert");
    expect(screen.queryByRole("button", { name: FREE })).toBeNull();
  });
});
