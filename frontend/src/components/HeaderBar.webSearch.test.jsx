// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// #310 — per-conversation Web Search toggle in the settings panel. Hidden by
// default so the Arena (which shares HeaderBar but has no conversation row)
// stays untouched; ConversationPage opts in with showWebSearch.

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

describe("HeaderBar web search toggle (#310)", () => {
  it("is hidden by default (Arena keeps the panel unchanged)", async () => {
    renderBar();
    openSettings();
    await screen.findAllByRole("slider");
    expect(screen.queryByRole("switch", { name: "Web search" })).toBeNull();
  });

  it("renders when showWebSearch is set, reflecting initialWebSearch", async () => {
    renderBar({ showWebSearch: true, initialWebSearch: true });
    openSettings();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    expect(toggle.getAttribute("aria-checked")).toBe("true");
  });

  it("defaults to off", async () => {
    renderBar({ showWebSearch: true });
    openSettings();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    expect(toggle.getAttribute("aria-checked")).toBe("false");
  });

  it("reports the flipped value through onWebSearchChange", async () => {
    const onWebSearchChange = vi.fn();
    renderBar({ showWebSearch: true, initialWebSearch: false, onWebSearchChange });
    openSettings();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    fireEvent.click(toggle);
    expect(onWebSearchChange).toHaveBeenCalledWith(true);
    expect(toggle.getAttribute("aria-checked")).toBe("true");
  });

  it("syncs with a later initialWebSearch prop change (hydration)", async () => {
    const { rerender } = renderBar({ showWebSearch: true, initialWebSearch: false });
    openSettings();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    rerender(
      <HeaderBar onApply={() => {}} onCustomizePrompt={() => {}} showWebSearch initialWebSearch />
    );
    expect(toggle.getAttribute("aria-checked")).toBe("true");
  });
});

// #570 — when the current model is positively known unable to execute tools,
// the toggle stays visible (so the state is not hidden) but is disabled and
// explains why, instead of silently doing nothing when flipped on.
describe("HeaderBar web search toggle disabled for tool-incapable models (#570)", () => {
  it("is enabled by default (webSearchDisabled unset)", async () => {
    renderBar({ showWebSearch: true, initialWebSearch: false });
    openSettings();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    expect(toggle.disabled).toBe(false);
  });

  it("is disabled when webSearchDisabled is set", async () => {
    renderBar({ showWebSearch: true, initialWebSearch: true, webSearchDisabled: true });
    openSettings();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    expect(toggle.disabled).toBe(true);
  });

  it("does not report a flip while disabled", async () => {
    const onWebSearchChange = vi.fn();
    renderBar({
      showWebSearch: true,
      initialWebSearch: false,
      webSearchDisabled: true,
      onWebSearchChange,
    });
    openSettings();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    fireEvent.click(toggle);
    expect(onWebSearchChange).not.toHaveBeenCalled();
  });

  it("keeps the toggle enabled when webSearchDisabled is explicitly false", async () => {
    renderBar({ showWebSearch: true, initialWebSearch: false, webSearchDisabled: false });
    openSettings();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    expect(toggle.disabled).toBe(false);
  });

  it("renders the caller-supplied, already-translated tooltip explaining why", async () => {
    renderBar({
      showWebSearch: true,
      initialWebSearch: false,
      webSearchDisabled: true,
      webSearchDisabledTooltip: "This model can't use tools, so web search isn't available for it.",
    });
    openSettings();
    await screen.findByRole("switch", { name: "Web search" });
    expect(document.body.textContent).toMatch(/can't use tools/i);
  });
});
