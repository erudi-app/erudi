// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import HelpTip from "./HelpTip";

afterEach(cleanup);

// The `?` next to each popup field (#585): a real button, so Tab reaches it,
// described by the help text it shows on hover and on focus.
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
});
