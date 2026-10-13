// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// The amber memory warning and the reload state:
//  - a `memory_warning` stream event renders an amber alert near the composer,
//    its copy chosen by `reason`: `bound` quotes the approximate memory the
//    model and the conversation take together (`footprint_bytes`); `pressure`
//    says the conversation is now summarized earlier; `pressure_external`
//    says other apps cause the pressure;
//  - the warning is transient state only: it never lands in the answer bubble
//    (so it cannot be copied with the message) and a LATER turn that carries
//    no warning clears it;
//  - a `memory_reload` event (the memory brake unloaded the model; with
//    `reason: "idle"`, the idle timeout did, and the copy says so) shows a
//    reload state in the pending assistant bubble -- even below a notice
//    already streamed as answer text -- until the model's first thinking,
//    answer or tool event, an error, or the end of the turn.

const { tracedFetchMock, navigateMock, locationMock, paramsMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
  locationMock: { pathname: "/conversation/7", state: null },
  paramsMock: { id: "7" },
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn(async () => []) },
  apiClient: { get: vi.fn(async () => []) },
  tracedFetch: tracedFetchMock,
}));

vi.mock("react-router-dom", () => ({
  useParams: () => ({ id: paramsMock.id }),
  useNavigate: () => navigateMock,
  useLocation: () => locationMock,
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/HeaderBar", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/TraceStrip", () => ({ default: () => null }));
vi.mock("../components/modals/CustomizePromptModal", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({
  default: ({ content }) => <div data-testid="answer">{content}</div>,
}));
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => <button onClick={() => onSend("hi", [], [])}>SEND</button>,
}));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";

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

/** Routes /query calls to a queue of controlled streams (one per turn). */
const makeRoute = (queryStreams) => {
  const pending = [...queryStreams];
  return async (url) => {
    const u = String(url);
    if (u.includes("/query")) return pending.shift().response;
    if (u.includes("generate_title")) return doneStream();
    if (/\/conversations\/\d+$/.test(u)) {
      return { ok: true, json: async () => conversationDetail };
    }
    if (u.includes("fetch_messages")) return { ok: true, json: async () => [] };
    return { ok: true, json: async () => [] };
  };
};

const renderAndSettle = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
  await screen.findByText("SEND");
};

const runTurn = async (stream, lines) => {
  await act(async () => {
    for (const line of lines) {
      stream.push(line);
    }
    stream.end();
  });
};

beforeEach(() => {
  vi.clearAllMocks();
  paramsMock.id = "7";
  Element.prototype.scrollTo = () => {};
});

afterEach(() => {
  cleanup();
});

describe("ConversationPage memory warning", () => {
  it("renders the amber warning with the conversation size, out of the bubble", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [
      '{"t":"answer","text":"Fine."}\n',
      '{"t":"memory_warning","reason":"bound","footprint_bytes":4294967296}\n',
      '{"t":"done"}\n',
    ]);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain(
      "Your computer's memory limits this conversation: the model and this conversation " +
        "together take about 4.0 GB, nearly all the room memory allows."
    );
    // Never part of the message content (so never in the copy path).
    for (const bubble of screen.getAllByTestId("answer")) {
      expect(bubble.textContent).not.toContain("4.0 GB");
      expect(bubble.textContent).toBe("Fine.");
    }
  });

  it("clears when a later turn carries no warning", async () => {
    const first = makeControlledStream();
    const second = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([first, second]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(first, [
      '{"t":"answer","text":"Long."}\n',
      '{"t":"memory_warning","reason":"bound","footprint_bytes":1048576}\n',
      '{"t":"done"}\n',
    ]);
    await screen.findByRole("alert");

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(second, ['{"t":"answer","text":"Compacted."}\n', '{"t":"done"}\n']);

    await waitFor(() => {
      expect(screen.queryByRole("alert")).toBeNull();
    });
  });

  it("clears when switching to another conversation", async () => {
    // [M2] The warning is about ONE conversation on THIS machine right now:
    // conversation A's warning must never show over conversation B.
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    const view = render(<ConversationPage />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    await act(async () => {});
    await screen.findByText("SEND");

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [
      '{"t":"answer","text":"Long."}\n',
      '{"t":"memory_warning","reason":"bound","footprint_bytes":1048576}\n',
      '{"t":"done"}\n',
    ]);
    await screen.findByRole("alert");

    paramsMock.id = "8";
    view.rerender(<ConversationPage />);

    await waitFor(() => {
      expect(screen.queryByRole("alert")).toBeNull();
    });
  });

  it.each([
    [
      "pressure",
      "Your computer's memory is under pressure: this conversation is summarized earlier to keep things fluid.",
    ],
    [
      "pressure_external",
      "Your computer's memory is under pressure from other apps; this conversation uses little of it.",
    ],
  ])("renders the %s copy, with no size to quote", async (reason, copy) => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [
      '{"t":"answer","text":"Fine."}\n',
      `{"t":"memory_warning","reason":"${reason}"}\n`,
      '{"t":"done"}\n',
    ]);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toBe(copy);
  });
});

describe("ConversationPage memory warning reasons", () => {
  it.each(["constructor", "toString", "unknown"])(
    "shows nothing for a reason it does not know (%s)",
    async (reason) => {
      const turn = makeControlledStream();
      tracedFetchMock.mockImplementation(makeRoute([turn]));
      await renderAndSettle();

      fireEvent.click(screen.getByText("SEND"));
      await runTurn(turn, [
        '{"t":"answer","text":"Fine."}\n',
        `{"t":"memory_warning","reason":"${reason}"}\n`,
        '{"t":"done"}\n',
      ]);

      await waitFor(() => expect(screen.getByTestId("answer").textContent).toBe("Fine."));
      expect(screen.queryByRole("alert")).toBeNull();
    }
  );
});

const RELOAD = "Memory was freed on your computer; reloading the model to resume the conversation…";
const IDLE_RELOAD =
  "The model was unloaded after a few minutes without use; reloading it to resume the conversation…";

describe("ConversationPage reload state", () => {
  const push = async (stream, line) => {
    await act(async () => {
      stream.push(line);
    });
  };

  it("shows while the model reloads and ends at its first answer", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await push(turn, '{"t":"memory_reload"}\n');
    expect(await screen.findByText(RELOAD)).toBeTruthy();

    await push(turn, '{"t":"answer","text":"Back."}\n');
    await waitFor(() => expect(screen.queryByText(RELOAD)).toBeNull());
    await act(async () => {
      turn.end();
    });
  });

  it("says the model was unloaded after a pause when the reason is idle", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await push(turn, '{"t":"memory_reload","reason":"idle"}\n');
    expect(await screen.findByText(IDLE_RELOAD)).toBeTruthy();
    expect(screen.queryByText(RELOAD)).toBeNull();

    await push(turn, '{"t":"answer","text":"Back."}\n');
    await waitFor(() => expect(screen.queryByText(IDLE_RELOAD)).toBeNull());
    await act(async () => {
      turn.end();
    });
  });

  it("keeps the memory copy for a reload without the idle reason", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await push(turn, '{"t":"memory_reload"}\n');
    expect(await screen.findByText(RELOAD)).toBeTruthy();
    expect(screen.queryByText(IDLE_RELOAD)).toBeNull();
    await act(async () => {
      turn.end();
    });
  });

  it("stays below a notice already streamed as answer text", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await push(turn, '{"t":"answer","text":"An image was ignored."}\n');
    await push(turn, '{"t":"memory_reload"}\n');
    expect(await screen.findByText(RELOAD)).toBeTruthy();
    expect(screen.getByTestId("answer").textContent).toBe("An image was ignored.");

    await push(turn, '{"t":"thinking","text":"hm"}\n');
    await waitFor(() => expect(screen.queryByText(RELOAD)).toBeNull());
    await act(async () => {
      turn.end();
    });
  });

  it.each([
    ["an error", '{"t":"error","text":"[ERROR_MESSAGE_SYSTEM] The model could not load."}\n'],
    ["the end of the turn", '{"t":"done"}\n'],
  ])("ends on %s", async (_label, line) => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await push(turn, '{"t":"memory_reload"}\n');
    expect(await screen.findByText(RELOAD)).toBeTruthy();

    await push(turn, line);
    await waitFor(() => expect(screen.queryByText(RELOAD)).toBeNull());
    await act(async () => {
      turn.end();
    });
  });

  it("never lands in the message content", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [
      '{"t":"memory_reload"}\n',
      '{"t":"answer","text":"Back."}\n',
      '{"t":"done"}\n',
    ]);

    await waitFor(() => expect(screen.getByTestId("answer").textContent).toBe("Back."));
    expect(screen.queryByText(RELOAD)).toBeNull();
  });
});
