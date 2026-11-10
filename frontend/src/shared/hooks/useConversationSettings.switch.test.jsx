// @vitest-environment jsdom
import React from "react";
import PropTypes from "prop-types";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act } from "@testing-library/react";

const { patchMock, putMock, getMock } = vi.hoisted(() => ({
  patchMock: vi.fn(),
  putMock: vi.fn(),
  getMock: vi.fn(),
}));
vi.mock("../../services/api/client", () => ({
  default: { patch: patchMock, put: putMock, get: getMock },
}));

import ConfigPopup from "../../components/modals/ConfigPopup.jsx";
import useConversationSettings from "./useConversationSettings";

const conversation = (id) => ({
  id,
  llm_id: 1,
  reasoning_effort: "medium",
  temperature: 0.7,
  top_p: 0.8,
  compaction_threshold: 80,
  web_search_enabled: false,
  custom_prompt: `prompt of ${id}`,
});
const MODEL = { id: 1, name: "M", supports_tools: true, supports_tools_wire: true };

// The conversation page: the open command popup closes in the same commit
// that changes the id.
function Page({ id }) {
  const settings = useConversationSettings(id);
  const [open, setOpen] = React.useState(false);
  const [seenId, setSeenId] = React.useState(id);
  if (seenId !== id) {
    setSeenId(id);
    setOpen(false);
  }
  return (
    <>
      <button data-testid="opener" onClick={() => setOpen(true)}>
        o
      </button>
      <button data-testid="adopt" onClick={() => settings.adopt(conversation(id))}>
        a
      </button>
      <ConfigPopup isOpen={open} onClose={() => setOpen(false)} settings={settings} model={MODEL} />
    </>
  );
}

Page.propTypes = { id: PropTypes.number.isRequired };

beforeEach(() => {
  patchMock.mockReset();
  getMock.mockReset();
  patchMock.mockImplementation(async (_path, body) => body);
});
afterEach(cleanup);

describe("useConversationSettings: switching conversation", () => {
  it("never saves an unsaved instructions edit onto the conversation switched to", async () => {
    const { rerender } = render(<Page id={7} />);
    fireEvent.click(screen.getByTestId("adopt"));
    const opener = screen.getByTestId("opener");
    opener.focus();
    fireEvent.click(opener);
    await act(async () => {});
    const box = screen.getByRole("textbox", { name: "Custom instructions" });
    box.focus();
    fireEvent.change(box, { target: { value: "edit made in 7" } });

    // The id changes with the popup still focused: closing it blurs the editor.
    rerender(<Page id={8} />);
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 400));
    });

    const toEight = patchMock.mock.calls.filter(([path]) => path === "/conversations/8");
    expect(toEight).toEqual([]);
  });
});
