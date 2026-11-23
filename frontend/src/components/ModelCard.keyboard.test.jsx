// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// The re-bind list of an orphaned assistant is a keyboard-operable menu, and
// the delete button carries an aria-label (#766).

vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: vi.fn() }),
}));

import ModelCard from "./ModelCard.jsx";

const orphan = {
  id: 12,
  name: "Support Assistant",
  kb_id: 3,
  weights_available: false,
  link: "org/base-model",
};
const targets = [
  { id: 1, name: "Base A" },
  { id: 2, name: "Base B" },
];

afterEach(cleanup);

const targetButtons = () => screen.getAllByRole("button", { name: /^Base / });

const setup = () => {
  const onRebind = vi.fn();
  render(<ModelCard model={orphan} type="local" rebindTargets={targets} onRebind={onRebind} />);
  return { onRebind, trigger: screen.getByRole("button", { name: /Re-bind/ }) };
};

describe("ModelCard re-bind menu keyboard access (#766)", () => {
  it("announces its expanded state and the list it controls", async () => {
    const user = userEvent.setup();
    const { trigger } = setup();
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    trigger.focus();
    await user.keyboard("{Enter}");
    expect(trigger.getAttribute("aria-expanded")).toBe("true");
    const list = document.getElementById(trigger.getAttribute("aria-controls"));
    expect(list).toBeTruthy();
    expect(list.textContent).toContain("Base A");
  });

  it("reaches each target with Tab and picks one with Enter, closing the menu", async () => {
    const user = userEvent.setup();
    const { onRebind, trigger } = setup();
    trigger.focus();
    await user.keyboard("{Enter}");
    const items = targetButtons();
    expect(items.map((i) => i.tagName)).toEqual(["BUTTON", "BUTTON"]);
    await user.tab();
    expect(document.activeElement).toBe(items[0]);
    await user.tab();
    expect(document.activeElement).toBe(items[1]);
    await user.keyboard("{Enter}");
    expect(onRebind).toHaveBeenCalledWith(orphan, targets[1]);
    expect(screen.queryByText("Base A")).toBeNull();
  });

  it("closes on Escape pressed on the trigger itself, focus staying there", async () => {
    const user = userEvent.setup();
    const { trigger } = setup();
    trigger.focus();
    await user.keyboard("{Enter}");
    await user.keyboard("{Escape}");
    expect(screen.queryByText("Base A")).toBeNull();
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
    expect(screen.queryByText("Base A")).toBeNull();
  });

  it("closes on a click outside", async () => {
    const user = userEvent.setup();
    const { trigger } = setup();
    await user.click(trigger);
    expect(screen.getByText("Base A")).toBeTruthy();
    await user.click(document.body);
    expect(screen.queryByText("Base A")).toBeNull();
  });

  it("closes on Escape and returns the focus to the trigger", async () => {
    const user = userEvent.setup();
    const { trigger } = setup();
    trigger.focus();
    await user.keyboard("{Enter}");
    await user.tab();
    await user.keyboard("{Escape}");
    expect(screen.queryByText("Base A")).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });
});

describe("ModelCard delete button", () => {
  it("has an aria-label", () => {
    render(<ModelCard model={{ name: "Qwen3 4B", size: "1 GB" }} type="local" />);
    expect(screen.getByRole("button", { name: "Delete model" }).getAttribute("aria-label")).toBe(
      "Delete model"
    );
  });
});
