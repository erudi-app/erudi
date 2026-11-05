// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act, within } from "@testing-library/react";

// /config (#609): every adjustable value of the conversation in one popup,
// each saved as it changes through the shared settings hook (no Apply). The
// popup is driven here with the real hook so the requests it makes are the
// ones the app makes.

const { patchMock, putMock, getMock } = vi.hoisted(() => ({
  patchMock: vi.fn(),
  putMock: vi.fn(),
  getMock: vi.fn(),
}));

vi.mock("../../services/api/client", () => ({
  default: { patch: patchMock, put: putMock, get: getMock },
}));

import ConfigPopup from "./ConfigPopup.jsx";
import ConfigGlobalPopup from "./ConfigGlobalPopup.jsx";
import EffortPopup from "./EffortPopup.jsx";
import useConversationSettings from "../../shared/hooks/useConversationSettings";

const CONVERSATION = {
  id: 7,
  llm_id: 1,
  reasoning_effort: "medium",
  temperature: 0.7,
  top_p: 0.8,
  compaction_threshold: 80,
  web_search_enabled: false,
  custom_prompt: "Be brief",
};

const TOOL_MODEL = {
  id: 1,
  name: "Qwen3-4B",
  supports_tools: true,
  supports_tools_wire: true,
  sampling_defaults: { temperature: 0.6, top_p: 0.95, source: "base_generation_config" },
};

let hookRef;

function Harness({ Popup = ConfigPopup, model = TOOL_MODEL, conversation = CONVERSATION }) {
  const settings = useConversationSettings(7);
  hookRef = settings;
  React.useEffect(() => {
    settings.adopt(conversation);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return <Popup isOpen onClose={() => {}} settings={settings} model={model} />;
}

beforeEach(() => {
  patchMock.mockReset();
  putMock.mockReset();
  getMock.mockReset();
  patchMock.mockImplementation(async (_p, body) => ({ ...CONVERSATION, ...body }));
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

const settle = () => act(async () => {});

describe("ConfigPopup", () => {
  it("shows every value of the conversation, with a subtitle, no Apply and no Max Tokens", async () => {
    render(<Harness />);
    await settle();
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("Conversation settings")).toBeTruthy();
    expect(
      within(dialog).getByText(
        "Changes are saved as you make them and apply from the next message."
      )
    ).toBeTruthy();
    expect(
      screen.getByRole("slider", { name: "Reasoning effort" }).getAttribute("aria-valuetext")
    ).toBe("Medium");
    expect(screen.getByRole("switch", { name: "Web search" }).getAttribute("aria-checked")).toBe(
      "false"
    );
    expect(screen.getByRole("slider", { name: "Creativity" }).value).toBe("0.7");
    expect(screen.getByRole("slider", { name: "Creativity" }).max).toBe("2");
    expect(screen.getByRole("slider", { name: "Diversity" }).value).toBe("0.8");
    expect(screen.getByRole("slider", { name: "Diversity" }).max).toBe("1");
    expect(screen.getByRole("combobox", { name: "Compaction threshold" }).value).toBe("80");
    expect(screen.getByRole("textbox", { name: "Custom instructions" }).value).toBe("Be brief");
    expect(screen.queryByRole("button", { name: /apply/i })).toBeNull();
    expect(screen.queryByText(/max tokens/i)).toBeNull();
  });

  it("carries a ? help button next to each field", async () => {
    render(<Harness />);
    await settle();
    for (const field of [
      "Reasoning effort",
      "Web search",
      "Creativity",
      "Diversity",
      "Compaction threshold",
      "Custom instructions",
    ]) {
      expect(screen.getByRole("button", { name: `Help: ${field}` })).toBeTruthy();
    }
  });

  it("offers 50 % to 90 % in steps of 5", async () => {
    render(<Harness />);
    await settle();
    const values = Array.from(
      screen.getByRole("combobox", { name: "Compaction threshold" }).options
    ).map((o) => Number(o.value));
    expect(values).toEqual([50, 55, 60, 65, 70, 75, 80, 85, 90]);
  });

  it("labels the five effort notches", async () => {
    render(<Harness />);
    await settle();
    for (const label of ["None", "Low", "Medium", "High", "Extra high"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
  });

  it("saves the effort, the switch and the threshold at once", async () => {
    render(<Harness />);
    await settle();
    fireEvent.change(screen.getByRole("slider", { name: "Reasoning effort" }), {
      target: { value: "4" },
    });
    fireEvent.click(screen.getByRole("switch", { name: "Web search" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Compaction threshold" }), {
      target: { value: "65" },
    });
    await act(async () => hookRef.flush());
    expect(patchMock.mock.calls.map((c) => c[1])).toEqual([
      { reasoning_effort: "xhigh" },
      { web_search_enabled: true },
      { compaction_threshold: 65 },
    ]);
  });

  it("saves the sliders 300 ms after they stop", async () => {
    vi.useFakeTimers();
    render(<Harness />);
    await settle();
    fireEvent.change(screen.getByRole("slider", { name: "Creativity" }), {
      target: { value: "1.5" },
    });
    fireEvent.change(screen.getByRole("slider", { name: "Diversity" }), {
      target: { value: "0.4" },
    });
    expect(patchMock).not.toHaveBeenCalled();
    await act(async () => {
      vi.advanceTimersByTime(300);
    });
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { temperature: 1.5, top_p: 0.4 });
  });

  it("saves the instructions on Save, an empty text included", async () => {
    render(<Harness />);
    await settle();
    const editor = screen.getByRole("textbox", { name: "Custom instructions" });
    fireEvent.change(editor, { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Save instructions" }));
    await act(async () => hookRef.flush());
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { custom_prompt: "" });
  });

  it("puts a control back and says so when its save fails", async () => {
    patchMock.mockRejectedValueOnce(new Error("offline"));
    render(<Harness />);
    await settle();
    fireEvent.change(screen.getByRole("combobox", { name: "Compaction threshold" }), {
      target: { value: "50" },
    });
    await act(async () => hookRef.flush());
    expect(screen.getByRole("combobox", { name: "Compaction threshold" }).value).toBe("80");
    expect(screen.getByRole("alert").textContent).toBe("Your change could not be saved.");
  });

  it("disables web search for a model that cannot use tools, and its help says why", async () => {
    render(<Harness model={{ ...TOOL_MODEL, supports_tools: false }} />);
    await settle();
    expect(screen.getByRole("switch", { name: "Web search" }).disabled).toBe(true);
    const help = screen.getByRole("button", { name: "Help: Web search" });
    expect(document.getElementById(help.getAttribute("aria-describedby")).textContent).toBe(
      "This model can't use tools, so web search isn't available for it."
    );
  });

  it("disables web search for a model whose tool calls are not verified yet", async () => {
    render(<Harness model={{ ...TOOL_MODEL, supports_tools_wire: null }} />);
    await settle();
    expect(screen.getByRole("switch", { name: "Web search" }).disabled).toBe(true);
    const help = screen.getByRole("button", { name: "Help: Web search" });
    expect(document.getElementById(help.getAttribute("aria-describedby")).textContent).toBe(
      "This model hasn't been verified to run tools yet, so web search stays off for it."
    );
  });

  // #570: the exact backend gate (kb_mode.py) is supports_tools AND
  // supports_tools_wire === true; an unknown model (orphaned) claims nothing.
  it.each([
    ["both flags true", { supports_tools: true, supports_tools_wire: true }, false],
    ["supports_tools false", { supports_tools: false, supports_tools_wire: true }, true],
    ["wire false", { supports_tools: true, supports_tools_wire: false }, true],
    ["wire unknown", { supports_tools: true, supports_tools_wire: null }, true],
    ["tools unknown", { supports_tools: null, supports_tools_wire: null }, true],
  ])("web search with %s: disabled = %s", async (_label, flags, disabled) => {
    render(<Harness model={{ ...TOOL_MODEL, ...flags }} />);
    await settle();
    expect(screen.getByRole("switch", { name: "Web search" }).disabled).toBe(disabled);
  });

  it("keeps web search enabled when the model is unknown (orphaned)", async () => {
    render(<Harness model={null} />);
    await settle();
    expect(screen.getByRole("switch", { name: "Web search" }).disabled).toBe(false);
  });

  it("says when the publisher recommends no sampling, and only then", async () => {
    render(
      <Harness
        model={{
          ...TOOL_MODEL,
          sampling_defaults: { temperature: 0.2, top_p: 0.95, source: "none" },
        }}
      />
    );
    await settle();
    expect(
      screen.getByText(
        "No sampling recommendation from this model's publisher; neutral defaults applied."
      )
    ).toBeTruthy();
    cleanup();
    render(<Harness />);
    await settle();
    expect(screen.queryByText(/No sampling recommendation/)).toBeNull();
  });
});

describe("EffortPopup", () => {
  it("is the effort slider alone, saved at once", async () => {
    render(<Harness Popup={EffortPopup} />);
    await settle();
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getAllByRole("slider")).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Help: Reasoning effort" })).toBeTruthy();
    fireEvent.change(screen.getByRole("slider", { name: "Reasoning effort" }), {
      target: { value: "0" },
    });
    await act(async () => hookRef.flush());
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { reasoning_effort: "none" });
  });
});

describe("ConfigGlobalPopup (/config global)", () => {
  it("saves the six values for the current model and names it", async () => {
    putMock.mockResolvedValue({});
    render(
      <Harness Popup={ConfigGlobalPopup} conversation={{ ...CONVERSATION, custom_prompt: "" }} />
    );
    await settle();
    await settle();
    expect(putMock).toHaveBeenCalledTimes(1);
    expect(putMock).toHaveBeenCalledWith("/llms/1/conversation_defaults", {
      reasoning_effort: "medium",
      temperature: 0.7,
      top_p: 0.8,
      compaction_threshold: 80,
      web_search_enabled: false,
      custom_prompt: "",
    });
    expect(screen.getByText("Saved as defaults for Qwen3-4B.")).toBeTruthy();
  });

  it("says when the defaults could not be saved, and retries", async () => {
    putMock.mockRejectedValueOnce(new Error("offline"));
    putMock.mockResolvedValueOnce({});
    render(<Harness Popup={ConfigGlobalPopup} />);
    await settle();
    await settle();
    expect(screen.getByRole("alert").textContent).toBe(
      "The defaults for Qwen3-4B could not be saved."
    );
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await settle();
    expect(putMock).toHaveBeenCalledTimes(2);
    expect(screen.getByText("Saved as defaults for Qwen3-4B.")).toBeTruthy();
  });
});
