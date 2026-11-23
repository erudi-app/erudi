// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// The Arena model dropdown is a keyboard-operable menu (#766).

globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

import HeaderBar from "./HeaderBar.jsx";

afterEach(cleanup);

const MODELS = [
  { id: 1, name: "Alpha" },
  { id: 2, name: "Beta" },
];

const entries = () =>
  screen.getAllByRole("button", { name: /^(Alpha|Beta)$/ }).filter((b) => b.closest("[id]"));

const setup = () => {
  const onModelChange = vi.fn();
  render(
    <HeaderBar
      onApply={() => {}}
      onLiveChange={() => {}}
      onCustomizePrompt={() => {}}
      models={MODELS}
      currentModel="Alpha"
      onModelChange={onModelChange}
    />
  );
  return { onModelChange, trigger: screen.getByRole("button", { name: "Select model" }) };
};

describe("HeaderBar model dropdown keyboard access (#766)", () => {
  it("is a button announcing its expanded state", () => {
    const { trigger } = setup();
    expect(trigger.tagName).toBe("BUTTON");
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
  });

  it("opens with Enter, picks an entry with Enter, and closes", async () => {
    const user = userEvent.setup();
    const { trigger, onModelChange } = setup();
    trigger.focus();
    await user.keyboard("{Enter}");
    const items = entries();
    expect(items.map((i) => i.tagName)).toEqual(["BUTTON", "BUTTON"]);
    await user.tab();
    await user.tab();
    await user.keyboard("{Enter}");
    expect(onModelChange).toHaveBeenCalledWith("Beta");
    expect(screen.queryByText("Beta")).toBeNull();
  });

  it("closes on Escape pressed on the button itself", async () => {
    const user = userEvent.setup();
    const { trigger } = setup();
    trigger.focus();
    await user.keyboard("{Enter}");
    expect(screen.getByText("Beta")).toBeTruthy();
    await user.keyboard("{Escape}");
    expect(screen.queryByText("Beta")).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it("closes when Tab leaves the list", async () => {
    const user = userEvent.setup();
    const { trigger } = setup();
    trigger.focus();
    await user.keyboard("{Enter}");
    await user.tab();
    await user.tab();
    await user.tab();
    expect(screen.queryByText("Beta")).toBeNull();
  });

  it("closes on Escape and refocuses the trigger", async () => {
    const user = userEvent.setup();
    const { trigger } = setup();
    trigger.focus();
    await user.keyboard("{Enter}");
    await user.tab();
    await user.keyboard("{Escape}");
    expect(screen.queryByText("Beta")).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });
});
