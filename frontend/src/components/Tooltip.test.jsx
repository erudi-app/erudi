// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import Tooltip from "./Tooltip";

afterEach(cleanup);

describe("Tooltip", () => {
  it("wraps its trigger and renders the content for hover display", () => {
    render(
      <Tooltip content="More info">
        <button>Trigger</button>
      </Tooltip>
    );
    expect(screen.getByText("Trigger")).toBeTruthy();
    const bubble = screen.getByText("More info").parentElement;
    expect(bubble.className).toContain("group-hover:opacity-100");
  });

  it.each([
    ["top", "bottom-full"],
    ["bottom", "top-full"],
    ["left", "right-full"],
    ["right", "left-full"],
    ["top-left", "bottom-full"],
    ["bottom-right", "top-full"],
  ])("positions the bubble for side=%s", (side, expectedClass) => {
    render(
      <Tooltip content={`tip-${side}`} side={side}>
        <span>t</span>
      </Tooltip>
    );
    const positioned = screen.getByText(`tip-${side}`).parentElement;
    expect(positioned.className).toContain(expectedClass);
  });

  it("falls back to the right-side placement for an unknown side", () => {
    render(
      <Tooltip content="fallback" side="diagonal" width="w-32">
        <span>t</span>
      </Tooltip>
    );
    const positioned = screen.getByText("fallback").parentElement;
    expect(positioned.className).toContain("left-full");
    expect(positioned.className).toContain("w-32");
  });
});

describe("Tooltip on keyboard focus (#585)", () => {
  it("shows the bubble while its trigger holds the focus, as on hover", () => {
    render(
      <Tooltip content="Focus info">
        <button>Trigger</button>
      </Tooltip>
    );
    const positioned = screen.getByText("Focus info").parentElement;
    expect(positioned.className).toContain("group-focus-within:opacity-100");
  });

  it("names the bubble a tooltip, with the id the trigger points at", () => {
    render(
      <Tooltip content="Described" id="tip-1">
        <button aria-describedby="tip-1">Trigger</button>
      </Tooltip>
    );
    const tooltip = screen.getByRole("tooltip");
    expect(tooltip.id).toBe("tip-1");
    expect(tooltip.textContent).toContain("Described");
  });
});
