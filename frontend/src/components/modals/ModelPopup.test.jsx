// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// /model (#585): the installed models, a model whose weights are missing not
// offered, the current one marked; picking one hands it to the page.

import ModelPopup from "./ModelPopup.jsx";

afterEach(cleanup);

const MODELS = [
  { id: 1, name: "Qwen3-4B" },
  { id: 2, name: "Gemma-3-4B" },
  { id: 3, name: "Orphan assistant", weights_available: false },
];

describe("ModelPopup", () => {
  it("lists the installed models with weights, the current one marked", () => {
    render(<ModelPopup isOpen onClose={() => {}} models={MODELS} currentId={1} onPick={vi.fn()} />);
    expect(screen.getByRole("button", { name: /Qwen3-4B/ }).getAttribute("aria-current")).toBe(
      "true"
    );
    expect(screen.getByRole("button", { name: /Gemma-3-4B/ })).toBeTruthy();
    expect(screen.queryByText("Orphan assistant")).toBeNull();
    expect(screen.getByRole("button", { name: "Help: Installed models" })).toBeTruthy();
  });

  it("hands the picked model to the page and closes", () => {
    const onPick = vi.fn();
    const onClose = vi.fn();
    render(<ModelPopup isOpen onClose={onClose} models={MODELS} currentId={1} onPick={onPick} />);
    fireEvent.click(screen.getByRole("button", { name: /Gemma-3-4B/ }));
    expect(onPick).toHaveBeenCalledWith(MODELS[1]);
    expect(onClose).toHaveBeenCalled();
  });

  it("closes without a request when the current model is picked again", () => {
    const onPick = vi.fn();
    const onClose = vi.fn();
    render(<ModelPopup isOpen onClose={onClose} models={MODELS} currentId={1} onPick={onPick} />);
    fireEvent.click(screen.getByRole("button", { name: /Qwen3-4B/ }));
    expect(onPick).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it("says so when no model can be offered", () => {
    render(
      <ModelPopup
        isOpen
        onClose={() => {}}
        models={[MODELS[2]]}
        currentId={null}
        onPick={vi.fn()}
      />
    );
    expect(screen.getByText("No installed model can be used right now.")).toBeTruthy();
  });
});
