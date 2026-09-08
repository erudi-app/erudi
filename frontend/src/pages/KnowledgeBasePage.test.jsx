// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// KnowledgeBasePage drives the embedding-model gate (#146) and the assistant
// creation form. These tests pin: the GATE.* state machine as the user sees it
// (mount check, resume-in-flight polling, download POST, error retry, leave
// navigation), the creation flow (validation, exact task payload with
// deduplicated dropped paths, success reset, error surfacing), the hardware
// readout fallbacks, and the URL-driven model preselection.

const { getMock, postMock, openKB, kbState, navigateMock, routerState } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  openKB: vi.fn(),
  kbState: { isCreating: false, isStarting: false },
  navigateMock: vi.fn(),
  routerState: { params: new URLSearchParams() },
}));

vi.mock("../services/api/client", () => ({
  default: { get: getMock, post: postMock },
  apiClient: { get: getMock, post: postMock },
}));

vi.mock("../contexts/KnowledgeBaseContext", () => ({
  useKnowledgeBase: () => ({
    open: openKB,
    isCreating: kbState.isCreating,
    isStarting: kbState.isStarting,
  }),
}));

vi.mock("react-router-dom", () => ({
  useSearchParams: () => [routerState.params],
  useNavigate: () => navigateMock,
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));

// Probe stubs: expose the props the page wires into its children so the tests
// can both observe state (selected model, model list size) and drive callbacks
// (model pick, file drop, gate actions) without the heavy real components.
/* eslint-disable react/prop-types -- the stubs below are test doubles; their
   props are pinned by the assertions, not by a PropTypes declaration. */

// The DragDropArea stub keeps state of its own, exactly like the real
// component does (the staged file list). That is what makes the reset
// assertion meaningful: clearing the parent's state does not touch it, so
// only a remount drops it. ModelLibrary owns no state of its own (#510):
// every name change goes straight to the parent via onModelNameChange, so
// the stub does the same instead of buffering a "locked" name locally.
vi.mock("../components/ModelLibrary", () => {
  const ModelLibraryStub = ({
    models,
    selectedModel,
    modelName,
    onModelSelect,
    onModelNameChange,
    onRefresh,
  }) => {
    return (
      <div>
        <span data-testid="lib-selected">{String(selectedModel)}</span>
        <span data-testid="lib-name">{modelName}</span>
        <span data-testid="lib-count">{models.length}</span>
        <button onClick={() => onModelSelect(5)}>PICK_MODEL</button>
        <button onClick={() => onModelNameChange("picked-name")}>SET_NAME</button>
        <button onClick={() => onRefresh()}>REFRESH_MODELS</button>
      </div>
    );
  };
  return { default: ModelLibraryStub };
});

vi.mock("../components/DragDropArea", () => {
  const DragDropAreaStub = ({ onFilesAdded }) => {
    const [staged, setStaged] = React.useState(0);
    return (
      <div>
        <span data-testid="dd-staged">{staged}</span>
        <button
          onClick={() => {
            setStaged(2);
            onFilesAdded([{ path: "/docs/a.pdf" }, "/docs/b.pdf", { path: "/docs/a.pdf" }]);
          }}
        >
          DROP_FILES
        </button>
      </div>
    );
  };
  return { default: DragDropAreaStub };
});

vi.mock("../components/modals/EmbeddingModelGateModal", () => ({
  default: ({ state, error, onDownload, onLeave, onClose }) => (
    <div data-testid="gate" data-state={state} data-error={error || ""}>
      <button onClick={onDownload}>GATE_DOWNLOAD</button>
      <button onClick={onLeave}>GATE_LEAVE</button>
      <button onClick={onClose}>GATE_CLOSE</button>
    </div>
  ),
}));

import KnowledgeBasePage from "./KnowledgeBasePage";
import i18n from "../i18n";

let hwResponder;
let modelsResponder;
let statusResponder;

const statusCalls = () =>
  getMock.mock.calls.filter(([p]) => p === "/knowledge_base/embedding-model/status");

const gateState = () => screen.getByTestId("gate").getAttribute("data-state");

beforeEach(() => {
  vi.spyOn(console, "log").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});

  getMock.mockReset();
  postMock.mockReset();
  openKB.mockReset();
  navigateMock.mockReset();
  kbState.isCreating = false;
  kbState.isStarting = false;
  routerState.params = new URLSearchParams();

  hwResponder = () => ({ global_inference_score: 91, global_inference_label: "Amazing" });
  modelsResponder = () => [];
  statusResponder = () => ({ available: true });

  getMock.mockImplementation(async (path) => {
    if (path === "/hardware/app_startup") return hwResponder();
    if (path === "/llms/local") return modelsResponder();
    if (path === "/knowledge_base/embedding-model/status") return statusResponder();
    return {};
  });
  postMock.mockResolvedValue({});
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("KnowledgeBasePage embedding-model gate", () => {
  it("shows no gate when the model is already on disk", async () => {
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(statusCalls()).toHaveLength(1));
    expect(screen.queryByTestId("gate")).toBeNull();
  });

  it("prompts to download when the model is absent", async () => {
    statusResponder = () => ({ available: false });
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(screen.getByTestId("gate")).toBeTruthy());
    expect(gateState()).toBe("prompt");
  });

  it("resumes an in-flight download, polls to done, and stops polling on close", async () => {
    vi.useFakeTimers();
    const statuses = [{ available: false, downloading: true }, { available: true }];
    statusResponder = () => (statuses.length > 1 ? statuses.shift() : statuses[0]);

    render(<KnowledgeBasePage />);
    await act(async () => {});
    expect(gateState()).toBe("downloading");

    // One poll tick later the model landed: "done" only ever follows an
    // active download, so the success screen (not a silent hide) is shown.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(gateState()).toBe("done");
    const callsAtDone = statusCalls().length;

    fireEvent.click(screen.getByText("GATE_CLOSE"));
    expect(screen.queryByTestId("gate")).toBeNull();

    // Leaving the downloading state must tear the interval down.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(statusCalls()).toHaveLength(callsAtDone);
  });

  it("enters the spinner and POSTs the download request on accept", async () => {
    statusResponder = () => ({ available: false });
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(screen.getByTestId("gate")).toBeTruthy());

    fireEvent.click(screen.getByText("GATE_DOWNLOAD"));
    expect(gateState()).toBe("downloading");
    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/knowledge_base/embedding-model/download")
    );
  });

  it("surfaces a failed download request as a retryable error", async () => {
    statusResponder = () => ({ available: false });
    postMock.mockRejectedValue(new Error("disk full"));
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(screen.getByTestId("gate")).toBeTruthy());

    fireEvent.click(screen.getByText("GATE_DOWNLOAD"));
    await waitFor(() => expect(gateState()).toBe("error"));
    expect(screen.getByTestId("gate").getAttribute("data-error")).toContain("disk full");
  });

  it("reports the backend-side error carried by the status payload", async () => {
    statusResponder = () => ({ available: false, error: "download crashed" });
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(screen.getByTestId("gate")).toBeTruthy());
    expect(gateState()).toBe("error");
    expect(screen.getByTestId("gate").getAttribute("data-error")).toBe("download crashed");
  });

  it("navigates back to the model library when the user declines", async () => {
    statusResponder = () => ({ available: false });
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(screen.getByTestId("gate")).toBeTruthy());

    fireEvent.click(screen.getByText("GATE_LEAVE"));
    expect(navigateMock).toHaveBeenCalledWith("/erudi/models");
  });

  it("keeps the page usable (no modal) when the status endpoint is unreachable", async () => {
    statusResponder = () => {
      throw new Error("backend down");
    };
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(statusCalls().length).toBeGreaterThan(0));
    expect(screen.queryByTestId("gate")).toBeNull();
  });
});

describe("KnowledgeBasePage assistant creation", () => {
  it("blocks submission until model, name and files are present", async () => {
    render(<KnowledgeBasePage />);
    fireEvent.click(screen.getByText("Create Assistant"));

    expect(await screen.findByText("Please fill in all required fields")).toBeTruthy();
    expect(openKB).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText("Close"));
    await waitFor(() =>
      expect(screen.queryByText("Please fill in all required fields")).toBeNull()
    );
  });

  it("submits the trimmed task with deduplicated dropped paths", async () => {
    render(<KnowledgeBasePage />);
    fireEvent.click(screen.getByText("PICK_MODEL"));
    fireEvent.click(screen.getByText("SET_NAME"));
    fireEvent.click(screen.getByText("DROP_FILES"));
    fireEvent.change(screen.getByPlaceholderText("Write a description"), {
      target: { value: "  my helper  " },
    });

    fireEvent.click(screen.getByText("Create Assistant"));

    expect(openKB).toHaveBeenCalledTimes(1);
    expect(openKB.mock.calls[0][0]).toEqual({
      paths: ["/docs/a.pdf", "/docs/b.pdf"],
      selectedModel: 5,
      modelName: "picked-name",
      description: "my helper",
      isUpdate: false,
    });
  });

  it("submits the name exactly as reported via onModelNameChange, with no separate lock step (#510)", async () => {
    // #510: ModelLibrary used to buffer a typed name locally until a
    // checkmark button "locked" it, so clicking Create Assistant without
    // that click submitted the parent's own pre-fill instead — silently
    // the pre-selected base model's own name. onModelNameChange is now the
    // only path a name takes from the input to the parent, so what it
    // reports is exactly what submitTrainForm sends and what the
    // duplicate-name guard checks, with nothing else able to override it.
    modelsResponder = () => [{ id: 5, name: "some-other-model" }];
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(screen.getByTestId("lib-count").textContent).toBe("1"));

    fireEvent.click(screen.getByText("PICK_MODEL"));
    fireEvent.click(screen.getByText("SET_NAME")); // calls onModelNameChange("picked-name") directly
    fireEvent.click(screen.getByText("DROP_FILES"));
    fireEvent.click(screen.getByText("Create Assistant"));

    expect(openKB).toHaveBeenCalledTimes(1);
    expect(openKB.mock.calls[0][0]).toMatchObject({ modelName: "picked-name" });
  });

  it("rejects a name already carried by another local model (#317)", async () => {
    modelsResponder = () => [
      { id: 5, name: "Qwen3 4B" },
      { id: 7, name: "picked-name" },
    ];
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(screen.getByTestId("lib-count").textContent).toBe("2"));

    fireEvent.click(screen.getByText("PICK_MODEL"));
    fireEvent.click(screen.getByText("SET_NAME"));
    fireEvent.click(screen.getByText("DROP_FILES"));
    fireEvent.click(screen.getByText("Create Assistant"));

    expect(await screen.findByText(/already exists/)).toBeTruthy();
    expect(openKB).not.toHaveBeenCalled();
  });

  it("treats the duplicate check case-insensitively (#317)", async () => {
    modelsResponder = () => [{ id: 5, name: "PICKED-NAME" }];
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(screen.getByTestId("lib-count").textContent).toBe("1"));

    fireEvent.click(screen.getByText("PICK_MODEL"));
    fireEvent.click(screen.getByText("SET_NAME"));
    fireEvent.click(screen.getByText("DROP_FILES"));
    fireEvent.click(screen.getByText("Create Assistant"));

    expect(await screen.findByText(/already exists/)).toBeTruthy();
    expect(openKB).not.toHaveBeenCalled();
  });

  it("flags the task as an update when the selected model is the assistant itself (#317)", async () => {
    // Updating an assistant keeps its own (necessarily existing) name: no
    // duplicate error, and the confirmation must say update, not create.
    modelsResponder = () => [{ id: 5, name: "picked-name", is_attached_to_kb: true, kb_id: 1 }];
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(screen.getByTestId("lib-count").textContent).toBe("1"));

    fireEvent.click(screen.getByText("PICK_MODEL"));
    fireEvent.click(screen.getByText("SET_NAME"));
    fireEvent.click(screen.getByText("DROP_FILES"));
    fireEvent.click(screen.getByText("Create Assistant"));

    expect(openKB).toHaveBeenCalledTimes(1);
    expect(openKB.mock.calls[0][0]).toMatchObject({
      selectedModel: 5,
      modelName: "picked-name",
      isUpdate: true,
    });
  });

  it("shows the success message then resets the form after the delay", async () => {
    vi.useFakeTimers();
    render(<KnowledgeBasePage />);
    await act(async () => {});
    fireEvent.click(screen.getByText("PICK_MODEL"));
    fireEvent.click(screen.getByText("SET_NAME"));
    fireEvent.click(screen.getByText("DROP_FILES"));
    fireEvent.click(screen.getByText("Create Assistant"));

    const opts = openKB.mock.calls[0][1];
    act(() => opts.onComplete());

    expect(screen.getByText("Data attached to your Assistant successfully!")).toBeTruthy();
    expect(screen.queryByText("Create Assistant")).toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(screen.getByText("Create Assistant")).toBeTruthy();
    expect(screen.getByTestId("lib-name").textContent).toBe("");
    expect(screen.getByPlaceholderText("Write a description").value).toBe("");
  });

  it("clears the staged file list the DragDropArea owns, and the submitted name, on reset", async () => {
    vi.useFakeTimers();
    render(<KnowledgeBasePage />);
    await act(async () => {});
    fireEvent.click(screen.getByText("PICK_MODEL"));
    fireEvent.click(screen.getByText("SET_NAME"));
    fireEvent.click(screen.getByText("DROP_FILES"));
    expect(screen.getByTestId("dd-staged").textContent).toBe("2");
    expect(screen.getByTestId("lib-name").textContent).toBe("picked-name");

    fireEvent.click(screen.getByText("Create Assistant"));
    act(() => openKB.mock.calls[0][1].onComplete());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });

    // Without the remount the page kept listing files it no longer holds, so
    // a second submission failed with "Please fill in all required fields"
    // under a visible file list. The name comes back to the parent's own
    // reset state (no child-owned copy left to remount away).
    expect(screen.getByTestId("dd-staged").textContent).toBe("0");
    expect(screen.getByTestId("lib-name").textContent).toBe("");
  });

  it("routes a creation error into the error modal", async () => {
    render(<KnowledgeBasePage />);
    fireEvent.click(screen.getByText("PICK_MODEL"));
    fireEvent.click(screen.getByText("SET_NAME"));
    fireEvent.click(screen.getByText("DROP_FILES"));
    fireEvent.click(screen.getByText("Create Assistant"));

    const opts = openKB.mock.calls[0][1];
    act(() => opts.onError("ingestion exploded"));

    expect(await screen.findByText("ingestion exploded")).toBeTruthy();
  });

  it("disables the button and relabels it while creating", async () => {
    kbState.isCreating = true;
    render(<KnowledgeBasePage />);
    const button = screen.getByText("Creating Assistant...");
    expect(button.disabled).toBe(true);
  });
});

describe("KnowledgeBasePage hardware readout", () => {
  it("renders the inference label and score chip once fetched", async () => {
    render(<KnowledgeBasePage />);
    expect(await screen.findByText("Amazing")).toBeTruthy();
    expect(screen.getByText("91/100")).toBeTruthy();
  });

  it("falls back to N/A when the backend omits score and label", async () => {
    hwResponder = () => ({});
    render(<KnowledgeBasePage />);
    expect((await screen.findAllByText("N/A")).length).toBeGreaterThan(0);
  });

  it("keeps mid-tier labels rendered (orange rating branch)", async () => {
    hwResponder = () => ({ global_inference_score: 55, global_inference_label: "Good" });
    render(<KnowledgeBasePage />);
    expect(await screen.findByText("Good")).toBeTruthy();
    expect(screen.getByText("55/100")).toBeTruthy();
  });

  it("shows the error placeholders when the hardware fetch fails", async () => {
    hwResponder = () => {
      throw new Error("hw down");
    };
    render(<KnowledgeBasePage />);
    expect((await screen.findAllByText("Error fetching")).length).toBeGreaterThan(0);
  });

  // The backend label is an English tier name; the page must show the same
  // translated tier the Models page shows, never the raw label (#387).
  it("translates the backend tier label like the Models page does", async () => {
    hwResponder = () => ({ global_inference_score: 52.62, global_inference_label: "Fair" });
    render(<KnowledgeBasePage />);
    expect(await screen.findByText("Fair")).toBeTruthy();
    cleanup();

    await i18n.changeLanguage("fr");
    try {
      render(<KnowledgeBasePage />);
      expect(await screen.findByText("Correct")).toBeTruthy();
      expect(screen.queryByText("Fair")).toBeNull();
    } finally {
      await i18n.changeLanguage("en");
    }
  });

  // The score was interpolated raw ("52.62/100" whatever the language); it
  // goes through the locale number formatter like every other number shown.
  it("formats the score in the active locale", async () => {
    hwResponder = () => ({ global_inference_score: 52.62, global_inference_label: "Fair" });
    render(<KnowledgeBasePage />);
    expect(await screen.findByText("52.6/100")).toBeTruthy();
    cleanup();

    await i18n.changeLanguage("fr");
    try {
      render(<KnowledgeBasePage />);
      expect(await screen.findByText("52,6/100")).toBeTruthy();
    } finally {
      await i18n.changeLanguage("en");
    }
  });
});

describe("KnowledgeBasePage model list and URL preselection", () => {
  it("preselects the model named in the URL, case-insensitively, but leaves a base model's name empty (#510)", async () => {
    // #510: pre-filling a base model's own name here is what let a name
    // typed afterwards but never confirmed silently submit as that base
    // model's name instead. A base model is never itself the name of the
    // new assistant, so the field the user sees (and can type into) starts
    // empty even though the model is selected.
    modelsResponder = () => [
      { id: 7, name: "Mistral" },
      { id: 8, name: "Qwen" },
    ];
    routerState.params = new URLSearchParams("model=mistral");
    render(<KnowledgeBasePage />);

    await waitFor(() => expect(screen.getByTestId("lib-selected").textContent).toBe("7"));
    expect(screen.getByTestId("lib-name").textContent).toBe("");
  });

  it("also matches the URL parameter against the model id, and still leaves the name empty for a base model", async () => {
    modelsResponder = () => [{ id: "abc", name: "X" }];
    routerState.params = new URLSearchParams("model=abc");
    render(<KnowledgeBasePage />);

    await waitFor(() => expect(screen.getByTestId("lib-selected").textContent).toBe("abc"));
    expect(screen.getByTestId("lib-name").textContent).toBe("");
  });

  it("pre-fills the name from the URL only when the found model is a KB assistant (#510)", async () => {
    // The update flow (#317) keeps the assistant's own name, so pre-filling
    // it here is safe and expected — unlike a base model's name.
    modelsResponder = () => [
      { id: 7, name: "Mistral" },
      { id: 9, name: "Support Helper", is_attached_to_kb: true, kb_id: 3 },
    ];
    routerState.params = new URLSearchParams("model=Support Helper");
    render(<KnowledgeBasePage />);

    await waitFor(() => expect(screen.getByTestId("lib-selected").textContent).toBe("9"));
    expect(screen.getByTestId("lib-name").textContent).toBe("Support Helper");
  });

  it("selects nothing when the URL model is unknown", async () => {
    modelsResponder = () => [{ id: 7, name: "Mistral" }];
    routerState.params = new URLSearchParams("model=ghost");
    render(<KnowledgeBasePage />);

    await waitFor(() => expect(screen.getByTestId("lib-count").textContent).toBe("1"));
    expect(screen.getByTestId("lib-selected").textContent).toBe("null");
    expect(screen.getByTestId("lib-name").textContent).toBe("");
  });

  it("recovers from a model list fetch failure with an empty library", async () => {
    modelsResponder = () => {
      throw new Error("llms down");
    };
    render(<KnowledgeBasePage />);
    await waitFor(() => expect(getMock.mock.calls.some(([p]) => p === "/llms/local")).toBe(true));
    expect(screen.getByTestId("lib-count").textContent).toBe("0");
  });

  it("refetches the model list when the library asks for a refresh", async () => {
    render(<KnowledgeBasePage />);
    const localCalls = () => getMock.mock.calls.filter(([p]) => p === "/llms/local");
    await waitFor(() => expect(localCalls()).toHaveLength(1));

    fireEvent.click(screen.getByText("REFRESH_MODELS"));
    await waitFor(() => expect(localCalls()).toHaveLength(2));
  });
});
