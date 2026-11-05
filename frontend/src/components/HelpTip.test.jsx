// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import HelpTip from "./HelpTip";
import ModalShell from "./modals/ModalShell";

afterEach(cleanup);

// The `?` next to each popup field (#585): a real button, so Tab reaches it,
// described by the help text it shows on hover and on focus. The text renders
// outside the popup (a portal on <body>, fixed position), so the popup's
// scrolling body never clips it. A closed tooltip is hidden from assistive
// technology; Escape on the `?` closes the tooltip alone (WCAG 1.4.13).
describe("HelpTip", () => {
  it("is a focusable button described by its help text", () => {
    render(<HelpTip label="Help: Creativity" text="Controls creativity." />);
    const button = screen.getByRole("button", { name: "Help: Creativity" });
    expect(button.tabIndex).toBe(0);
    const tooltip = screen.getByRole("tooltip", { hidden: true });
    expect(button.getAttribute("aria-describedby")).toBe(tooltip.id);
    expect(tooltip.textContent).toContain("Controls creativity.");
    button.focus();
    expect(document.activeElement).toBe(button);
  });

  it("renders its text outside any clipping container", () => {
    render(
      <div data-testid="clipper" style={{ overflow: "auto" }}>
        <HelpTip label="Help: Field" text="Some help" />
      </div>
    );
    const tooltip = screen.getByRole("tooltip", { hidden: true });
    expect(screen.getByTestId("clipper").contains(tooltip)).toBe(false);
    expect(tooltip.closest("[data-help-tooltip]").style.position).toBe("fixed");
  });

  it("is hidden from assistive technology while closed, shown on focus and on hover", () => {
    render(<HelpTip label="Help: Field" text="Some help" />);
    const button = screen.getByRole("button", { name: "Help: Field" });
    const layer = () =>
      screen.getByRole("tooltip", { hidden: true }).closest("[data-help-tooltip]");
    expect(screen.queryByRole("tooltip")).toBeNull();
    expect(layer().style.visibility).toBe("hidden");
    fireEvent.focus(button);
    expect(layer().dataset.open).toBe("true");
    expect(screen.getByRole("tooltip")).toBeTruthy();
    fireEvent.blur(button);
    expect(layer().dataset.open).toBe("false");
    fireEvent.mouseEnter(button);
    expect(layer().dataset.open).toBe("true");
    fireEvent.mouseLeave(button);
    expect(layer().dataset.open).toBe("false");
  });

  it("closes only its tooltip on Escape, not the popup holding it", () => {
    const onClose = vi.fn();
    render(
      <ModalShell isOpen onClose={onClose} title="T">
        <HelpTip label="Help: Field" text="Some help" />
        <input aria-label="field" />
      </ModalShell>
    );
    const button = screen.getByRole("button", { name: "Help: Field" });
    button.focus();
    fireEvent.focus(button);
    expect(screen.getByRole("tooltip")).toBeTruthy();
    fireEvent.keyDown(button, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).toBeNull();
    expect(onClose).not.toHaveBeenCalled();
    // A second Escape, the tooltip closed, closes the popup.
    fireEvent.keyDown(button, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
