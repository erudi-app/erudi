// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import HeaderBar from "./HeaderBar";

// HeaderBar observes its own width with ResizeObserver, which jsdom lacks.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver || ResizeObserverStub;

// When the model's publisher gives no sampling recommendation (#388,
// `sampling_defaults.source === "none"`), the settings panel says so in a
// discreet line under the sliders. When a recommendation exists, nothing new.

const NOTE = "No sampling recommendation from this model's publisher; neutral defaults applied.";

const openSettings = () => fireEvent.click(screen.getByLabelText("Toggle settings"));

afterEach(() => {
  cleanup();
});

describe("HeaderBar noPublisherRecommendation", () => {
  it("shows the muted note under the sliders when the prop is set", async () => {
    render(<HeaderBar noPublisherRecommendation />);
    openSettings();
    const note = await screen.findByTestId("no-publisher-recommendation");
    expect(note.textContent).toBe(NOTE);
    expect(note.tagName).toBe("P");
  });

  it("shows nothing without the prop (a recommendation exists or is unknown)", async () => {
    render(<HeaderBar />);
    openSettings();
    await screen.findAllByRole("slider");
    expect(screen.queryByTestId("no-publisher-recommendation")).toBeNull();
    expect(screen.queryByText(NOTE)).toBeNull();
  });

  it("is not rendered while the panel is closed", () => {
    render(<HeaderBar noPublisherRecommendation />);
    expect(screen.queryByTestId("no-publisher-recommendation")).toBeNull();
  });
});
