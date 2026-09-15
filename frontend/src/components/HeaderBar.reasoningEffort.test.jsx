// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// PR-D2 — per-conversation Reasoning effort control in the settings panel,
// next to the Web search toggle. Hidden by default so the Arena (which
// shares HeaderBar but has no conversation row -- its turns follow the
// GLOBAL default backend-side) keeps an unchanged panel; ConversationPage
// opts in with showReasoningEffort.

import HeaderBar from "./HeaderBar.jsx";

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver || ResizeObserverStub;

const renderBar = (props = {}) =>
  render(<HeaderBar onApply={() => {}} onCustomizePrompt={() => {}} {...props} />);

const openSettings = () => fireEvent.click(screen.getByLabelText("Toggle settings"));

afterEach(() => {
  cleanup();
});

describe("HeaderBar reasoning effort control (PR-D2)", () => {
  it("is hidden by default (Arena keeps the panel unchanged)", async () => {
    renderBar();
    openSettings();
    await screen.findAllByRole("slider");
    expect(screen.queryByRole("combobox", { name: "Reasoning effort" })).toBeNull();
  });

  it("renders when showReasoningEffort is set, with the five levels", async () => {
    renderBar({ showReasoningEffort: true, initialReasoningEffort: "medium" });
    openSettings();
    const select = await screen.findByRole("combobox", { name: "Reasoning effort" });
    const values = Array.from(select.querySelectorAll("option")).map((o) => o.value);
    expect(values).toEqual(["none", "low", "medium", "high", "xhigh"]);
  });

  it("reflects initialReasoningEffort", async () => {
    renderBar({ showReasoningEffort: true, initialReasoningEffort: "high" });
    openSettings();
    const select = await screen.findByRole("combobox", { name: "Reasoning effort" });
    expect(select.value).toBe("high");
  });

  it("reports the new value through onReasoningEffortChange", async () => {
    const onReasoningEffortChange = vi.fn();
    renderBar({
      showReasoningEffort: true,
      initialReasoningEffort: "medium",
      onReasoningEffortChange,
    });
    openSettings();
    const select = await screen.findByRole("combobox", { name: "Reasoning effort" });
    fireEvent.change(select, { target: { value: "xhigh" } });
    expect(onReasoningEffortChange).toHaveBeenCalledWith("xhigh");
    expect(select.value).toBe("xhigh");
  });

  it("syncs with a later initialReasoningEffort prop change (hydration)", async () => {
    const { rerender } = renderBar({
      showReasoningEffort: true,
      initialReasoningEffort: "low",
    });
    openSettings();
    const select = await screen.findByRole("combobox", { name: "Reasoning effort" });
    expect(select.value).toBe("low");
    rerender(
      <HeaderBar
        onApply={() => {}}
        onCustomizePrompt={() => {}}
        showReasoningEffort
        initialReasoningEffort="none"
      />
    );
    expect(select.value).toBe("none");
  });
});
