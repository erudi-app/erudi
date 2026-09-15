// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// #136 — the settings controls used to cap below what the backend accepts
// (temperature 0-2). The slider bounds now match, and live edits above the old
// caps still reach onLiveChange (#218).
//
// The panel carries no output-budget control anymore: the budget is derived
// backend-side from the context window the model runs in.

import HeaderBar from "./HeaderBar.jsx";

// HeaderBar observes its own width with ResizeObserver, which jsdom lacks.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver || ResizeObserverStub;

const renderBar = (props = {}) =>
  render(<HeaderBar onApply={() => {}} onCustomizePrompt={() => {}} {...props} />);

// The controls live behind the settings toggle; open it before querying them.
const openSettings = () => fireEvent.click(screen.getByLabelText("Toggle settings"));

afterEach(() => {
  cleanup();
});

describe("HeaderBar settings bounds (#136)", () => {
  it("caps the creativity slider at 2 and diversity at 1", async () => {
    renderBar();
    openSettings();

    const sliders = await screen.findAllByRole("slider");
    const [temperature, topP] = sliders;
    expect(temperature.getAttribute("max")).toBe("2");
    expect(temperature.getAttribute("min")).toBe("0");
    expect(temperature.getAttribute("step")).toBe("0.01");
    expect(topP.getAttribute("max")).toBe("1");
  });

  it("offers no output-budget control", async () => {
    renderBar();
    openSettings();

    await screen.findAllByRole("slider");
    expect(screen.queryByRole("spinbutton")).toBeNull();
  });

  it("emits temperature above the old 1.0 cap via onLiveChange", async () => {
    const onLiveChange = vi.fn();
    renderBar({ onLiveChange });
    openSettings();

    const [temperature] = await screen.findAllByRole("slider");
    fireEvent.change(temperature, { target: { value: "1.8" } });

    expect(onLiveChange).toHaveBeenLastCalledWith(expect.objectContaining({ temperature: 1.8 }));
    // The value badge reflects the above-1.0 value.
    expect(screen.getByText("1.80")).toBeTruthy();
  });
});
