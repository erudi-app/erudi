// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within, act } from "@testing-library/react";

// The command surface on a conversation (#585, #609): the static header, its
// settings panel, the slash commands typed in the composer, and the popups
// they open, wired to the conversation through one settings owner. The real
// header, composer and popups render here; only the network is faked.

const { tracedFetchMock, navigateMock, locationMock, paramsMock, getMock, patchMock, putMock } =
  vi.hoisted(() => ({
    tracedFetchMock: vi.fn(),
    navigateMock: vi.fn(),
    locationMock: { pathname: "/conversation/7", state: null },
    // The route's conversation id: a test moves it to switch conversation.
    paramsMock: { id: "7" },
    getMock: vi.fn(),
    patchMock: vi.fn(),
    putMock: vi.fn(),
  }));

vi.mock("../services/api/client", () => ({
  default: { get: getMock, patch: patchMock, put: putMock },
  tracedFetch: tracedFetchMock,
}));

vi.mock("react-router-dom", () => ({
  useParams: () => paramsMock,
  useNavigate: () => navigateMock,
  useLocation: () => locationMock,
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({
  default: ({ content }) => <div>{content}</div>,
}));
// The engine failure dialog, reduced to a marker.
vi.mock("../components/modals/EngineFailureModal", () => ({
  default: ({ notice }) => (notice ? <div data-testid="engine-failure">{notice.code}</div> : null),
}));

import ConversationPage from "./ConversationPage.jsx";

const MODELS = [
  { id: 1, name: "Qwen3-4B", supports_tools: true, supports_tools_wire: true },
  { id: 2, name: "Gemma-3-4B", supports_tools: true, supports_tools_wire: true },
  { id: 3, name: "Orphan assistant", weights_available: false },
];

const DETAIL = {
  id: 7,
  llm_id: 1,
  reasoning_effort: "medium",
  temperature: 0.7,
  top_p: 0.8,
  max_tokens: 512,
  compaction_threshold: 80,
  web_search_enabled: false,
  custom_prompt: "Be brief",
};

const encoder = new TextEncoder();
const answerStream = () => {
  const queue = [
    { done: false, value: encoder.encode('{"t":"answer","text":"ok"}\n{"t":"done"}\n') },
    { done: true, value: undefined },
  ];
  return { ok: true, body: { getReader: () => ({ read: async () => queue.shift() }) } };
};
/** A streaming response fed line by line: the answer stays open until end(). */
function controlledStream() {
  const queue = [];
  let waiting = null;
  let ended = false;
  const deliver = (chunk) => {
    if (waiting) {
      const resolve = waiting;
      waiting = null;
      resolve(chunk);
    } else {
      queue.push(chunk);
    }
  };
  return {
    response: {
      ok: true,
      body: {
        getReader: () => ({
          read: () =>
            new Promise((resolve) => {
              if (queue.length) resolve(queue.shift());
              else if (ended) resolve({ done: true, value: undefined });
              else waiting = resolve;
            }),
        }),
      },
    },
    push: (line) => deliver({ done: false, value: encoder.encode(line + "\n") }),
    end: () => {
      ended = true;
      deliver({ done: true, value: undefined });
    },
  };
}

const deferred = () => {
  let resolve;
  const promise = new Promise((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

const emptyStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

let detail;
let order;

const routeFetch = async (url, opts = {}) => {
  const u = String(url);
  if (u.includes("generate_title")) return emptyStream();
  if (u.includes("/query")) {
    order.push(["query", JSON.parse(opts.body)]);
    return answerStream();
  }
  if (u.endsWith("/llms/local")) return { ok: true, json: async () => MODELS };
  if (u.endsWith("/conversations/7")) return { ok: true, json: async () => detail };
  return { ok: true, json: async () => [] };
};

const composer = () => document.querySelector("textarea");
const typeAndEnter = (value) => {
  fireEvent.change(composer(), { target: { value } });
  fireEvent.keyDown(composer(), { key: "Enter" });
};
const header = () => screen.getByRole("heading", { level: 3 });
const queries = () => order.filter(([kind]) => kind === "query");

const renderPage = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(header().textContent).toContain("Qwen3-4B"));
};

beforeEach(() => {
  Element.prototype.scrollTo = () => {};
  detail = { ...DETAIL };
  order = [];
  locationMock.state = null;
  paramsMock.id = "7";
  getMock.mockReset();
  getMock.mockImplementation(async () => []);
  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(routeFetch);
  patchMock.mockReset();
  patchMock.mockImplementation(async (_path, body) => {
    order.push(["patch", body]);
    return { ...detail, ...body };
  });
  putMock.mockReset();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("the header", () => {
  it("reads a static line naming the conversation's model", async () => {
    await renderPage();
    expect(header().textContent).toBe("You are chatting with Qwen3-4B");
  });

  it("opens /config from its settings panel, showing the conversation's values", async () => {
    await renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
    fireEvent.click(screen.getByRole("button", { name: /^Config/ }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByRole("slider", { name: "Creativity" }).value).toBe("0.7");
    expect(within(dialog).getByRole("textbox", { name: "Custom instructions" }).value).toBe(
      "Be brief"
    );
  });

  it("offers to unload the loaded model from its settings panel", async () => {
    getMock.mockImplementation(async (path) =>
      path === "/llms/loaded" ? { llm_id: 9, name: "Another Model" } : []
    );
    await renderPage();
    expect(getMock).not.toHaveBeenCalledWith("/llms/loaded");
    fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
    expect(await screen.findByRole("button", { name: "Unload Another Model" })).toBeTruthy();
  });

  it("shows the Unload button once a turn has loaded the model, with the panel open", async () => {
    let loaded = null;
    getMock.mockImplementation(async (path) => (path === "/llms/loaded" ? loaded : []));
    await renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
    await waitFor(() => expect(getMock).toHaveBeenCalledWith("/llms/loaded"));
    expect(screen.queryByRole("button", { name: /^Unload/ })).toBeNull();

    loaded = { llm_id: 1, name: "Qwen3-4B" };
    typeAndEnter("Hello");
    expect(await screen.findByRole("button", { name: "Unload Qwen3-4B" })).toBeTruthy();
  });

  it("re-reads the loaded model once a reload turn has ended, not at its first event", async () => {
    // The reload notice is the turn's first event and comes before the model
    // is loaded: only the read after the turn ends can see the model.
    let loaded = null;
    getMock.mockImplementation(async (path) => (path === "/llms/loaded" ? loaded : []));
    const stream = controlledStream();
    tracedFetchMock.mockImplementation(async (url, opts) =>
      String(url).includes("/query") ? stream.response : routeFetch(url, opts)
    );
    const reads = () => getMock.mock.calls.filter(([path]) => path === "/llms/loaded").length;
    await renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
    await waitFor(() => expect(reads()).toBe(1));

    typeAndEnter("Hello again");
    await waitFor(() => expect(composer().disabled).toBe(true));
    await act(async () => {
      stream.push('{"t":"memory_reload","reason":"manual"}');
    });
    // Whatever the page read while the model was loading, it saw nothing.
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
    const readsDuringTheTurn = reads();
    expect(screen.queryByRole("button", { name: /^Unload/ })).toBeNull();

    loaded = { llm_id: 1, name: "Qwen3-4B" };
    await act(async () => {
      stream.push('{"t":"answer","text":"Back."}');
      stream.push('{"t":"done"}');
      stream.end();
    });
    expect(await screen.findByRole("button", { name: "Unload Qwen3-4B" })).toBeTruthy();
    expect(reads()).toBeGreaterThan(readsDuringTheTurn);
  });

  it("names the newly loaded model after a /model switch, with the panel open", async () => {
    let loaded = { llm_id: 1, name: "Qwen3-4B" };
    getMock.mockImplementation(async (path) => (path === "/llms/loaded" ? loaded : []));
    patchMock.mockImplementation(async () => ({ ...detail, llm_id: 2 }));
    await renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
    expect(await screen.findByRole("button", { name: "Unload Qwen3-4B" })).toBeTruthy();

    loaded = { llm_id: 2, name: "Gemma-3-4B" };
    typeAndEnter("/model");
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: /Gemma-3-4B/ }));
    expect(await screen.findByRole("button", { name: "Unload Gemma-3-4B" })).toBeTruthy();
  });

  it("shows no Unload button when no model is loaded", async () => {
    getMock.mockImplementation(async (path) => (path === "/llms/loaded" ? null : []));
    await renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
    await waitFor(() => expect(getMock).toHaveBeenCalledWith("/llms/loaded"));
    expect(screen.queryByRole("button", { name: /^Unload/ })).toBeNull();
  });
});

describe("slash commands in the composer", () => {
  it("/config opens the popup and sends nothing to the model", async () => {
    await renderPage();
    typeAndEnter("/config");
    expect(await screen.findByRole("dialog")).toBeTruthy();
    expect(queries()).toHaveLength(0);
    expect(composer().value).toBe("");
  });

  it("/model switches with a PATCH of llm_id alone and adopts the answer", async () => {
    await renderPage();
    patchMock.mockImplementation(async (_path, body) => {
      order.push(["patch", body]);
      return { ...detail, llm_id: 2, temperature: 0.3, top_p: 0.9, custom_prompt: "Gemma rules" };
    });
    typeAndEnter("/model");
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).queryByText("Orphan assistant")).toBeNull();
    fireEvent.click(within(dialog).getByRole("button", { name: /Gemma-3-4B/ }));

    await waitFor(() => expect(header().textContent).toBe("You are chatting with Gemma-3-4B"));
    expect(patchMock).toHaveBeenCalledTimes(1);
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { llm_id: 2 });

    typeAndEnter("Hello");
    await waitFor(() => expect(queries()).toHaveLength(1));
    expect(queries()[0][1]).toMatchObject({
      temperature: 0.3,
      top_p: 0.9,
      custom_prompt: "Gemma rules",
    });
  });

  it("the header's model name picks through the same PATCH as /model (#704)", async () => {
    await renderPage();
    patchMock.mockImplementation(async () => ({
      ...detail,
      llm_id: 2,
      temperature: 0.3,
      top_p: 0.9,
    }));
    fireEvent.click(screen.getByRole("button", { name: /Qwen3-4B/ }));
    const listbox = screen.getByRole("listbox");
    expect(within(listbox).queryByText("Orphan assistant")).toBeNull();
    fireEvent.click(within(listbox).getByRole("option", { name: "Gemma-3-4B" }));

    await waitFor(() => expect(header().textContent).toBe("You are chatting with Gemma-3-4B"));
    expect(patchMock).toHaveBeenCalledTimes(1);
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { llm_id: 2 });
  });

  it("the settings window's Save as defaults button saves like /config global", async () => {
    putMock.mockResolvedValue({});
    await renderPage();
    typeAndEnter("/config");
    fireEvent.click(await screen.findByRole("button", { name: "Save as defaults for Qwen3-4B" }));
    await screen.findByText("Saved as defaults for Qwen3-4B.");
    expect(putMock).toHaveBeenCalledWith("/llms/1/conversation_defaults", expect.any(Object));
  });

  it("/config global saves the conversation's values for its model", async () => {
    putMock.mockResolvedValue({});
    await renderPage();
    typeAndEnter("/config global");
    await screen.findByText("Saved as defaults for Qwen3-4B.");
    expect(putMock).toHaveBeenCalledWith("/llms/1/conversation_defaults", {
      reasoning_effort: "medium",
      temperature: 0.7,
      top_p: 0.8,
      compaction_threshold: 80,
      web_search_enabled: false,
      custom_prompt: "Be brief",
    });
  });
});

describe("sending right after a change", () => {
  it("sends no turn before the pending save has come back", async () => {
    await renderPage();
    const save = deferred();
    patchMock.mockImplementation((_path, body) => {
      order.push(["patch", body]);
      return save.promise.then(() => ({ ...detail, ...body }));
    });
    typeAndEnter("/effort");
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByRole("slider", { name: "Reasoning effort" }), {
      target: { value: "4" },
    });
    fireEvent.keyDown(dialog, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    typeAndEnter("Right away");
    await waitFor(() => expect(order).toHaveLength(1));
    // The effort is read server-side: the turn must not start before it is saved.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(tracedFetchMock.mock.calls.some(([u]) => String(u).includes("/query"))).toBe(false);

    save.resolve();
    await waitFor(() => expect(queries()).toHaveLength(1));
    expect(order.map(([kind]) => kind)).toEqual(["patch", "query"]);
  });

  it("waits for the pending slider save, then sends the new value with the turn", async () => {
    await renderPage();
    typeAndEnter("/config");
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByRole("slider", { name: "Creativity" }), {
      target: { value: "1.3" },
    });
    fireEvent.keyDown(dialog, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    typeAndEnter("Right away");
    await waitFor(() => expect(queries()).toHaveLength(1));
    expect(order.map(([kind]) => kind)).toEqual(["patch", "query"]);
    expect(order[0][1]).toEqual({ temperature: 1.3 });
    expect(queries()[0][1].temperature).toBe(1.3);
  });
});

describe("while an answer streams", () => {
  const streaming = async () => {
    const stream = controlledStream();
    tracedFetchMock.mockImplementation(async (url, opts) =>
      String(url).includes("/query") ? stream.response : routeFetch(url, opts)
    );
    await renderPage();
    typeAndEnter("A long answer please");
    await waitFor(() => expect(composer().disabled).toBe(true));
    return stream;
  };

  it("locks the header's model picker while the answer streams", async () => {
    const stream = await streaming();
    expect(screen.getByRole("button", { name: /Qwen3-4B/ }).disabled).toBe(true);
    stream.push('{"t":"done"}');
    stream.end();
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Qwen3-4B/ }).disabled).toBe(false)
    );
  });

  it("disables Model in the header panel, and keeps Config and Effort", async () => {
    const stream = await streaming();
    fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
    expect(screen.getByRole("button", { name: /^Model/ }).disabled).toBe(true);
    expect(screen.getByRole("button", { name: /^Config/ }).disabled).toBe(false);
    expect(screen.getByRole("button", { name: /^Effort/ }).disabled).toBe(false);
    stream.push('{"t":"done"}');
    stream.end();
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^Model/ }).disabled).toBe(false)
    );
  });

  it("closes an open command popup before the engine failure dialog opens", async () => {
    const stream = await streaming();
    fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
    fireEvent.click(screen.getByRole("button", { name: /^Config/ }));
    expect(await screen.findByRole("dialog")).toBeTruthy();

    stream.push('{"t":"error","text":"[ERROR_MESSAGE_SYSTEM] failed","code":"CUDA_OOM"}');
    stream.end();
    await screen.findByTestId("engine-failure");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });
});

describe("before the conversation's values are loaded", () => {
  const gatedDetail = () => {
    const gate = deferred();
    tracedFetchMock.mockImplementation(async (url, opts) => {
      if (String(url).endsWith("/conversations/7")) {
        await gate.promise;
      }
      return routeFetch(url, opts);
    });
    return gate;
  };

  it("keeps the composer disabled, so no turn goes out without its instructions", async () => {
    const gate = gatedDetail();
    render(<ConversationPage />);
    await waitFor(() => expect(composer()).toBeTruthy());
    expect(composer().disabled).toBe(true);
    gate.resolve();
    await waitFor(() => expect(composer().disabled).toBe(false));
  });

  it("drops a load that lands after /model switched the conversation", async () => {
    const gate = gatedDetail();
    patchMock.mockImplementation(async (_path, body) => ({ ...detail, ...body }));
    render(<ConversationPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Conversation settings" }));
    fireEvent.click(screen.getByRole("button", { name: /^Model/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Gemma-3-4B/ }));
    await waitFor(() => expect(header().textContent).toBe("You are chatting with Gemma-3-4B"));

    gate.resolve(); // the older row (llm_id 1) arrives now
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(header().textContent).toBe("You are chatting with Gemma-3-4B");
  });
});

describe("switching conversation while the previous one still loads", () => {
  it("never takes A's late load into B: header, messages and values stay B's", async () => {
    const gateA = deferred();
    const DETAIL_B = { ...DETAIL, id: 8, llm_id: 2, custom_prompt: "B's instructions" };
    const messages = {
      7: [{ id: 1, sender: "user", content: "A's message" }],
      8: [{ id: 2, sender: "user", content: "B's message" }],
    };
    const MESSAGES_PATH = /^\/conversations\/(\d+)\/fetch_messages$/;
    const CONVERSATION_URL = /\/conversations\/(\d+)(\/fetch_messages|\/query)?$/;
    getMock.mockImplementation(async (path) => {
      const match = path.match(MESSAGES_PATH);
      if (match?.[1] === "7") await gateA.promise;
      return match ? messages[match[1]] : [];
    });
    tracedFetchMock.mockImplementation(async (url, opts) => {
      const match = String(url).match(CONVERSATION_URL);
      if (match?.[2] === "/query") {
        order.push(["query", JSON.parse(opts.body)]);
        return answerStream();
      }
      if (match?.[1] === "7") {
        await gateA.promise;
        return { ok: true, json: async () => (match[2] ? messages[7] : DETAIL) };
      }
      if (match?.[1] === "8") {
        return { ok: true, json: async () => (match[2] ? messages[8] : DETAIL_B) };
      }
      return routeFetch(url, opts);
    });

    const { rerender } = render(<ConversationPage />);
    paramsMock.id = "8";
    rerender(<ConversationPage />);
    await waitFor(() => expect(header().textContent).toBe("You are chatting with Gemma-3-4B"));
    expect(screen.getByText("B's message")).toBeTruthy();

    gateA.resolve(); // A's load lands last
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(header().textContent).toBe("You are chatting with Gemma-3-4B");
    expect(screen.queryByText("A's message")).toBeNull();
    expect(screen.getByText("B's message")).toBeTruthy();

    typeAndEnter("Hello B");
    await waitFor(() => expect(queries()).toHaveLength(1));
    expect(queries()[0][1].custom_prompt).toBe("B's instructions");
  });
});

describe("when the conversation cannot be read", () => {
  const serverError = async () => ({ ok: false, status: 500, json: async () => ({}) });
  const networkError = async () => {
    throw new TypeError("Failed to fetch");
  };

  it.each([
    ["the server answers an error", serverError],
    ["the request fails", networkError],
  ])(
    "shows an error with a Retry when %s, the composer staying locked",
    async (_label, failing) => {
      let fail = true;
      tracedFetchMock.mockImplementation(async (url, opts) =>
        String(url).endsWith("/conversations/7") && fail ? failing() : routeFetch(url, opts)
      );
      render(<ConversationPage />);
      await screen.findByText("This conversation could not be loaded.");
      expect(composer().disabled).toBe(true);

      fail = false;
      fireEvent.click(screen.getByRole("button", { name: "Retry" }));
      await waitFor(() => expect(composer().disabled).toBe(false));
      expect(screen.queryByText("This conversation could not be loaded.")).toBeNull();
      expect(header().textContent).toBe("You are chatting with Qwen3-4B");
    }
  );
});

describe("an orphaned conversation", () => {
  it("unblocks sending when a model is picked from the red header text (#704)", async () => {
    detail = { ...DETAIL, llm_id: null };
    render(<ConversationPage />);
    const slot = await screen.findByRole("button", { name: "Please select a model" });
    expect(composer().disabled).toBe(true);
    patchMock.mockImplementation(async (_path, body) => ({ ...detail, ...body }));
    fireEvent.click(slot);
    fireEvent.click(screen.getByRole("option", { name: "Qwen3-4B" }));
    await waitFor(() => expect(composer().disabled).toBe(false));
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { llm_id: 1 });
  });

  it("keeps the Model button working while the composer is blocked", async () => {
    detail = { ...DETAIL, llm_id: null };
    render(<ConversationPage />);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Please select a model" }).textContent).toBe(
        "Please select a model"
      )
    );
    expect(composer().disabled).toBe(true);

    patchMock.mockImplementation(async (_path, body) => ({ ...detail, ...body }));
    fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
    fireEvent.click(screen.getByRole("button", { name: /^Model/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Qwen3-4B/ }));

    await waitFor(() => expect(composer().disabled).toBe(false));
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { llm_id: 1 });
    expect(header().textContent).toBe("You are chatting with Qwen3-4B");
  });

  it.each([
    ["missing from the installed list", 99],
    ["an assistant whose weights are missing", 3],
  ])("is blocked when its model is %s", async (_label, llmId) => {
    detail = { ...DETAIL, llm_id: llmId };
    render(<ConversationPage />);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Please select a model" }).textContent).toBe(
        "Please select a model"
      )
    );
    expect(composer().disabled).toBe(true);
  });
});

describe("loading the header (#217)", () => {
  it("names the model when the models list resolves after the conversation", async () => {
    let releaseModels;
    tracedFetchMock.mockImplementation(async (url, opts) => {
      if (String(url).endsWith("/llms/local")) {
        await new Promise((resolve) => {
          releaseModels = resolve;
        });
        return { ok: true, json: async () => MODELS };
      }
      return routeFetch(url, opts);
    });
    render(<ConversationPage />);
    await waitFor(() => expect(releaseModels).toBeTypeOf("function"));
    expect(screen.queryByRole("alert")).toBeNull();
    releaseModels();
    await waitFor(() => expect(header().textContent).toBe("You are chatting with Qwen3-4B"));
  });

  it("names the model when the conversation resolves after the models list", async () => {
    let releaseDetail;
    tracedFetchMock.mockImplementation(async (url, opts) => {
      if (String(url).endsWith("/conversations/7")) {
        await new Promise((resolve) => {
          releaseDetail = resolve;
        });
      }
      return routeFetch(url, opts);
    });
    render(<ConversationPage />);
    await waitFor(() => expect(releaseDetail).toBeTypeOf("function"));
    expect(screen.queryByRole("alert")).toBeNull();
    releaseDetail();
    await waitFor(() => expect(header().textContent).toBe("You are chatting with Qwen3-4B"));
  });
});

describe("/config follows the conversation's model", () => {
  it("shows the no-recommendation note once switched to a model without one", async () => {
    MODELS[1].sampling_defaults = { temperature: 0.2, top_p: 0.95, source: "none" };
    try {
      await renderPage();
      typeAndEnter("/config");
      let dialog = await screen.findByRole("dialog");
      expect(within(dialog).queryByText(/No sampling recommendation/)).toBeNull();
      fireEvent.keyDown(dialog, { key: "Escape" });
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

      typeAndEnter("/model");
      dialog = await screen.findByRole("dialog");
      fireEvent.click(within(dialog).getByRole("button", { name: /Gemma-3-4B/ }));
      await waitFor(() => expect(header().textContent).toBe("You are chatting with Gemma-3-4B"));

      typeAndEnter("/config");
      dialog = await screen.findByRole("dialog");
      expect(within(dialog).getByText(/No sampling recommendation/)).toBeTruthy();
    } finally {
      delete MODELS[1].sampling_defaults;
    }
  });
});

describe("the first message from the new-conversation page", () => {
  it("is sent with the values of the conversation as created", async () => {
    locationMock.state = { initialQuestion: "First question" };
    detail = { ...DETAIL, temperature: 0.4, top_p: 0.6, custom_prompt: "Saved for the model" };
    render(<ConversationPage />);
    await waitFor(() => expect(queries()).toHaveLength(1));
    expect(queries()[0][1]).toMatchObject({
      question: "First question",
      temperature: 0.4,
      top_p: 0.6,
      custom_prompt: "Saved for the model",
    });
  });
});
