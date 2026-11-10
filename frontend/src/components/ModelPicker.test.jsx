// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// The model name of the header as a picker (#704): one click, or the keyboard,
// opens the installed models; picking one hands it to the page, which applies
// it through the same path as the Model button.

import ModelPicker from "./ModelPicker.jsx";

afterEach(cleanup);

const MODELS = [
  { id: 1, name: "Qwen3-4B" },
  { id: 2, name: "Gemma-3-4B" },
  { id: 3, name: "Orphan assistant", weights_available: false },
];

const setup = (props = {}) => {
  const onPick = vi.fn();
  const utils = render(
    <ModelPicker modelName="Qwen3-4B" models={MODELS} currentId={1} onPick={onPick} {...props} />
  );
  return { onPick, ...utils };
};

const trigger = () => screen.getByRole("button", { name: /Qwen3-4B|Please select a model/ });

describe("ModelPicker", () => {
  it("shows the model name on a button with a chevron, closed", () => {
    setup();
    expect(trigger().getAttribute("aria-haspopup")).toBe("listbox");
    expect(trigger().getAttribute("aria-expanded")).toBe("false");
    expect(trigger().querySelector("svg")).toBeTruthy();
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("opens on click and lists the installed models, orphans left out", () => {
    setup();
    fireEvent.click(trigger());
    expect(trigger().getAttribute("aria-expanded")).toBe("true");
    const options = screen.getAllByRole("option").map((o) => o.textContent);
    expect(options).toEqual(["Qwen3-4BCurrent", "Gemma-3-4B"]);
    expect(screen.getByRole("option", { name: /Qwen3-4B/ }).getAttribute("aria-selected")).toBe(
      "true"
    );
    expect(screen.queryByText("Orphan assistant")).toBeNull();
  });

  it("picking a model calls onPick with it and closes", async () => {
    const user = userEvent.setup();
    const { onPick } = setup();
    await user.click(trigger());
    await user.click(screen.getByRole("option", { name: "Gemma-3-4B" }));
    expect(onPick).toHaveBeenCalledWith(MODELS[1]);
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("picking the current model only closes", async () => {
    const user = userEvent.setup();
    const { onPick } = setup();
    await user.click(trigger());
    await user.click(screen.getByRole("option", { name: /Qwen3-4B/ }));
    expect(onPick).not.toHaveBeenCalled();
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("opens with Enter and Space on the trigger", async () => {
    const user = userEvent.setup();
    setup();
    trigger().focus();
    await user.keyboard("{Enter}");
    expect(screen.getByRole("listbox")).toBeTruthy();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("listbox")).toBeNull();
    await user.keyboard(" ");
    expect(screen.getByRole("listbox")).toBeTruthy();
  });

  it("opens with the arrow keys on the trigger", async () => {
    const user = userEvent.setup();
    setup();
    trigger().focus();
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("listbox")).toBeTruthy();
  });

  it("moves with the arrows, picks with Enter", async () => {
    const user = userEvent.setup();
    const { onPick } = setup();
    trigger().focus();
    await user.keyboard("{Enter}");
    const listbox = screen.getByRole("listbox");
    expect(document.activeElement).toBe(listbox);
    const active = () => document.getElementById(listbox.getAttribute("aria-activedescendant"));
    expect(active().textContent).toContain("Qwen3-4B");
    await user.keyboard("{ArrowDown}");
    expect(active().textContent).toBe("Gemma-3-4B");
    await user.keyboard("{ArrowDown}");
    expect(active().textContent).toBe("Gemma-3-4B");
    await user.keyboard("{ArrowUp}{ArrowUp}");
    expect(active().textContent).toContain("Qwen3-4B");
    await user.keyboard("{End}");
    expect(active().textContent).toBe("Gemma-3-4B");
    await user.keyboard("{Enter}");
    expect(onPick).toHaveBeenCalledWith(MODELS[1]);
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(document.activeElement).toBe(trigger());
  });

  it("Escape closes and returns the focus to the trigger", async () => {
    const user = userEvent.setup();
    const { onPick } = setup();
    await user.click(trigger());
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(document.activeElement).toBe(trigger());
    expect(onPick).not.toHaveBeenCalled();
  });

  it("closes on a click outside", async () => {
    const user = userEvent.setup();
    setup();
    await user.click(trigger());
    await user.click(document.body);
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("cannot be opened while disabled, and closes if it becomes disabled", () => {
    const onPick = vi.fn();
    const props = { modelName: "Qwen3-4B", models: MODELS, currentId: 1, onPick };
    const { rerender } = render(<ModelPicker {...props} disabled />);
    expect(trigger().disabled).toBe(true);
    fireEvent.click(trigger());
    expect(screen.queryByRole("listbox")).toBeNull();
    rerender(<ModelPicker {...props} />);
    fireEvent.click(trigger());
    expect(screen.getByRole("listbox")).toBeTruthy();
    rerender(<ModelPicker {...props} disabled />);
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("orphaned conversation: red text in the name slot, opens the picker, picking unblocks", async () => {
    const user = userEvent.setup();
    const { onPick } = setup({ modelName: "", attention: true, currentId: 9 });
    const slot = screen.getByRole("button", { name: "Please select a model" });
    expect(slot.textContent).toBe("Please select a model");
    expect(slot.querySelector(".text-red-400")).toBeTruthy();
    await user.click(slot);
    await user.click(screen.getByRole("option", { name: "Gemma-3-4B" }));
    expect(onPick).toHaveBeenCalledWith(MODELS[1]);
  });

  it("says how to get a model when none can be offered", async () => {
    const user = userEvent.setup();
    setup({ models: [MODELS[2]], modelName: "", attention: true });
    await user.click(screen.getByRole("button", { name: "Please select a model" }));
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(screen.getByText(/Open Models to download one/)).toBeTruthy();
    await user.keyboard("{Escape}");
    expect(screen.queryByText(/Open Models to download one/)).toBeNull();
  });

  it("closes when the current model changes under it (another conversation)", () => {
    const onPick = vi.fn();
    const props = { modelName: "Qwen3-4B", models: MODELS, onPick };
    const { rerender } = render(<ModelPicker {...props} currentId={1} />);
    fireEvent.click(trigger());
    expect(screen.getByRole("listbox")).toBeTruthy();
    rerender(<ModelPicker {...props} currentId={2} />);
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("always names an existing option when the list shrinks while open", async () => {
    const user = userEvent.setup();
    const onPick = vi.fn();
    const props = { modelName: "Qwen3-4B", onPick, currentId: 1 };
    const { rerender } = render(<ModelPicker {...props} models={MODELS} />);
    await user.click(trigger());
    await user.keyboard("{ArrowDown}");
    rerender(<ModelPicker {...props} models={[MODELS[0]]} />);
    const listbox = screen.getByRole("listbox");
    expect(document.getElementById(listbox.getAttribute("aria-activedescendant"))).toBeTruthy();
  });

  it("moves the focus to the fallback element when it is locked while open", () => {
    const fallback = React.createRef();
    const props = { modelName: "Qwen3-4B", models: MODELS, currentId: 1, onPick: vi.fn() };
    const { rerender } = render(
      <>
        <button ref={fallback}>stable</button>
        <ModelPicker {...props} fallbackFocusRef={fallback} />
      </>
    );
    fireEvent.click(trigger());
    screen.getByRole("listbox").focus();
    rerender(
      <>
        <button ref={fallback}>stable</button>
        <ModelPicker {...props} fallbackFocusRef={fallback} disabled />
      </>
    );
    expect(document.activeElement).toBe(fallback.current);
  });

  it("points to Models, not to a re-bind, when nothing can be offered", () => {
    setup({ models: [MODELS[2]], modelName: "", currentId: null });
    fireEvent.click(screen.getByRole("button", { name: "Please select a model" }));
    expect(
      screen.getByText("No installed model can be used. Open Models to download one.")
    ).toBeTruthy();
  });
});
