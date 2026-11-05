// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import HelpTip from "./HelpTip";

afterEach(cleanup);

// The `?` next to each popup field (#585): a real button, so Tab reaches it,
// described by the help text it shows on hover and on focus. The text renders
// outside the popup (a portal on <body>, fixed position), so the popup's
// scrolling body never clips it.
describe("HelpTip", () => {
  it("is a focusable button described by its help text", () => {
    render(<HelpTip label="Help: Creativity" text="Controls creativity." />);
    const button = screen.getByRole("button", { name: "Help: Creativity" });
    expect(button.tabIndex).toBe(0);
    const tooltip = screen.getByRole("tooltip");
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
    const tooltip = screen.getByRole("tooltip");
    expect(screen.getByTestId("clipper").contains(tooltip)).toBe(false);
    expect(tooltip.closest("[data-help-tooltip]").style.position).toBe("fixed");
  });

  it("shows on keyboard focus and on hover, and hides again", () => {
    render(<HelpTip label="Help: Field" text="Some help" />);
    const button = screen.getByRole("button", { name: "Help: Field" });
    const layer = () => screen.getByRole("tooltip").closest("[data-help-tooltip]");
    expect(layer().dataset.open).toBe("false");
    fireEvent.focus(button);
    expect(layer().dataset.open).toBe("true");
    fireEvent.blur(button);
    expect(layer().dataset.open).toBe("false");
    fireEvent.mouseEnter(button);
    expect(layer().dataset.open).toBe("true");
    fireEvent.mouseLeave(button);
    expect(layer().dataset.open).toBe("false");
  });
});
