// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act } from "@testing-library/react";

// The Save as defaults button of the conversation's settings window (#704)
// does what /config global does: same request, same confirmation.

const { patchMock, putMock, getMock } = vi.hoisted(() => ({
  patchMock: vi.fn(),
  putMock: vi.fn(),
  getMock: vi.fn(),
}));
vi.mock("../services/api/client", () => ({
  default: { patch: patchMock, put: putMock, get: getMock },
}));

import CommandPopups from "./CommandPopups.jsx";
import useConversationSettings from "../shared/hooks/useConversationSettings";

const CONVERSATION = {
  id: 7,
  llm_id: 1,
  reasoning_effort: "medium",
  temperature: 0.7,
  top_p: 0.8,
  compaction_threshold: 80,
  web_search_enabled: false,
  custom_prompt: "",
};
const MODEL = { id: 1, name: "Qwen3-4B" };

function Harness({ initial, saveDefaultsDisabled = false }) {
  const settings = useConversationSettings(7);
  const [command, setCommand] = React.useState(initial);
  React.useEffect(() => {
    settings.adopt(CONVERSATION);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <CommandPopups
      command={command}
      onClose={() => setCommand(null)}
      onCommand={setCommand}
      saveDefaultsDisabled={saveDefaultsDisabled}
      settings={settings}
      model={MODEL}
      models={[MODEL]}
      onPickModel={vi.fn()}
    />
  );
}

beforeEach(() => {
  putMock.mockReset();
  putMock.mockResolvedValue({});
});
afterEach(cleanup);

const settle = () => act(async () => {});

describe("Save as defaults", () => {
  it("is at the bottom of /config, names the model, and saves like /config global", async () => {
    render(<Harness initial={{ name: "config", arg: null }} />);
    await settle();
    const button = screen.getByRole("button", { name: "Save as defaults for Qwen3-4B" });
    expect(putMock).not.toHaveBeenCalled();
    fireEvent.click(button);
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

  it("is disabled while an answer is being written, like /config global", async () => {
    render(<Harness initial={{ name: "config", arg: null }} saveDefaultsDisabled />);
    await settle();
    const button = screen.getByRole("button", { name: "Save as defaults for Qwen3-4B" });
    expect(button.disabled).toBe(true);
    fireEvent.click(button);
    await settle();
    expect(putMock).not.toHaveBeenCalled();
  });
});
