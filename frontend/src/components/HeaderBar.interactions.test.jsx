// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act, waitFor } from "@testing-library/react";

// Interactive surface of HeaderBar beyond the bounds already pinned in
// HeaderBar.test.jsx: the model dropdown (open/select/outside-close/disabled),
// the Apply commit, the diversity slider live edit, the picker-attention
// alert, and the width-tier responsive classes driven by ResizeObserver.

// Capture the ResizeObserver callback so tests can feed synthetic widths.
let resizeCallback = null;
class CapturingResizeObserver {
  constructor(cb) {
    resizeCallback = cb;
  }
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = CapturingResizeObserver;

import HeaderBar from "./HeaderBar.jsx";

const MODELS = [
  { id: 1, name: "Alpha" },
  { id: 2, name: "Beta" },
];

const renderBar = (props = {}) =>
  render(<HeaderBar onApply={() => {}} onCustomizePrompt={() => {}} {...props} />);

const openSettings = () => fireEvent.click(screen.getByLabelText("Toggle settings"));

afterEach(() => {
  cleanup();
  resizeCallback = null;
});

describe("HeaderBar model dropdown", () => {
  it("opens on click, reports the picked model and closes", () => {
    const onModelChange = vi.fn();
    renderBar({ models: MODELS, currentModel: "Alpha", onModelChange });

    fireEvent.click(screen.getByLabelText("Select model"));
    fireEvent.click(screen.getByText("Beta"));

    expect(onModelChange).toHaveBeenCalledWith("Beta");
    expect(screen.queryByText("Beta")).toBeNull(); // list closed
  });

  it("closes on an outside mousedown without changing the model", () => {
    const onModelChange = vi.fn();
    renderBar({ models: MODELS, currentModel: "Alpha", onModelChange });

    fireEvent.click(screen.getByLabelText("Select model"));
    expect(screen.getByText("Beta")).toBeTruthy();

    fireEvent.mouseDown(document.body);
    expect(screen.queryByText("Beta")).toBeNull();
    expect(onModelChange).not.toHaveBeenCalled();
  });

  it("does not open while the bar is disabled", () => {
    renderBar({ models: MODELS, currentModel: "Alpha", disabled: true });

    fireEvent.click(screen.getByLabelText("Select model"));
    expect(screen.queryByText("Beta")).toBeNull();
  });

  it("shows the placeholder when no model is selected and the attention alert when asked", () => {
    renderBar({
      models: MODELS,
      currentModel: "",
      pickerAttention: true,
      pickerAttentionMessage: "Pick a model to continue",
    });

    expect(screen.getByText("Select model...")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toBe("Pick a model to continue");
  });
});

describe("HeaderBar apply and live edits", () => {
  it("commits the tuned values through onApply and closes the panel", async () => {
    const onApply = vi.fn();
    renderBar({ onApply, initialTemperature: 0.3, initialTopP: 0.9 });
    openSettings();

    const [temperature, topP] = await screen.findAllByRole("slider");
    fireEvent.change(temperature, { target: { value: "0.7" } });
    fireEvent.change(topP, { target: { value: "0.4" } });

    fireEvent.click(screen.getByText("Apply"));

    expect(onApply).toHaveBeenCalledWith({ temperature: 0.7, topP: 0.4 });
    await waitFor(() => expect(screen.queryByText("Apply")).toBeNull());
  });

  it("pushes diversity edits live through onLiveChange", async () => {
    const onLiveChange = vi.fn();
    renderBar({ onLiveChange, initialTemperature: 0.2 });
    openSettings();

    const [, topP] = await screen.findAllByRole("slider");
    fireEvent.change(topP, { target: { value: "0.33" } });

    expect(onLiveChange).toHaveBeenLastCalledWith({ temperature: 0.2, topP: 0.33 });
    expect(screen.getByText("0.33")).toBeTruthy();
  });

  it("forwards the Customize Prompt click", async () => {
    const onCustomizePrompt = vi.fn();
    renderBar({ onCustomizePrompt });
    openSettings();

    fireEvent.click(await screen.findByText("Customize Prompt"));
    expect(onCustomizePrompt).toHaveBeenCalledTimes(1);
  });
});

describe("HeaderBar width tiers", () => {
  const rootOf = (container) => container.querySelector(".hb-scope");

  it("maps observed widths to the xs/sm/md/lg tier classes", () => {
    const { container } = renderBar();

    act(() => resizeCallback([{ contentRect: { width: 300 } }]));
    expect(rootOf(container).className).toContain("hb-xs");

    act(() => resizeCallback([{ contentRect: { width: 400 } }]));
    expect(rootOf(container).className).toContain("hb-sm");

    act(() => resizeCallback([{ contentRect: { width: 600 } }]));
    expect(rootOf(container).className).toContain("hb-md");

    act(() => resizeCallback([{ contentRect: { width: 800 } }]));
    expect(rootOf(container).className).toContain("hb-lg");
  });

  it("falls back to the element width when the entry has no contentRect", () => {
    const { container } = renderBar();

    // jsdom reports offsetWidth 0 -> the xs tier.
    act(() => resizeCallback([]));
    expect(rootOf(container).className).toContain("hb-xs");
  });
});
