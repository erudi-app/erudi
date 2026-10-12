// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

import CompactionMarker from "./CompactionMarker.jsx";

// Where a compaction took over (#614): one separator per `done` event, its copy
// chosen by the reason and the count, the summary readable on demand as plain
// text, and an honest line when the summary was lost.

const done = (overrides = {}) => ({
  t: "compaction",
  phase: "done",
  summarized: 12,
  reason: "length",
  summary: "Ada lives in Lyon.",
  summary_lost: false,
  ...overrides,
});

afterEach(() => {
  cleanup();
});

describe("CompactionMarker", () => {
  it.each([
    [
      "length",
      12,
      "12 earlier messages were summarized to keep the conversation within the model's context",
    ],
    [
      "length",
      1,
      "1 earlier message was summarized to keep the conversation within the model's context",
    ],
    [
      "length",
      0,
      "Earlier messages were summarized to keep the conversation within the model's context",
    ],
    ["memory", 7, "7 earlier messages were summarized to fit this computer's memory"],
    ["memory", 1, "1 earlier message was summarized to fit this computer's memory"],
    ["memory", 0, "Earlier messages were summarized to fit this computer's memory"],
  ])("says why (%s) and how many (%i)", (reason, summarized, copy) => {
    render(<CompactionMarker events={[done({ reason, summarized })]} />);

    expect(screen.getByText(copy)).toBeTruthy();
  });

  it("falls back to the length copy for a reason it does not know", () => {
    render(<CompactionMarker events={[done({ reason: "constructor", summarized: 2 })]} />);

    expect(
      screen.getByText(
        "2 earlier messages were summarized to keep the conversation within the model's context"
      )
    ).toBeTruthy();
  });

  it("shows the summary on demand, as plain text", () => {
    render(<CompactionMarker events={[done({ summary: "**Ada** lives in Lyon." })]} />);

    expect(screen.queryByText("**Ada** lives in Lyon.")).toBeNull();
    const toggle = screen.getByRole("button", { name: "Show summary" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");

    fireEvent.click(toggle);
    // Raw text, never markdown: the asterisks stay.
    expect(screen.getByText("**Ada** lives in Lyon.")).toBeTruthy();
    const hide = screen.getByRole("button", { name: "Hide summary" });
    expect(hide.getAttribute("aria-expanded")).toBe("true");

    fireEvent.click(hide);
    expect(screen.queryByText("**Ada** lives in Lyon.")).toBeNull();
  });

  it("stacks one separator per compaction, each with its own toggle", () => {
    render(
      <CompactionMarker
        events={[
          done({ summarized: 4, summary: "First." }),
          done({ summarized: 2, summary: "Second." }),
        ]}
      />
    );

    const lines = screen.getAllByText(/earlier messages were summarized/);
    expect(lines.map((l) => l.textContent)).toEqual([
      "4 earlier messages were summarized to keep the conversation within the model's context",
      "2 earlier messages were summarized to keep the conversation within the model's context",
    ]);
    fireEvent.click(screen.getAllByRole("button", { name: "Show summary" })[1]);
    expect(screen.getByText("Second.")).toBeTruthy();
    expect(screen.queryByText("First.")).toBeNull();
  });

  it("says a lost summary was dropped, and offers the summary the model keeps", () => {
    render(
      <CompactionMarker events={[done({ summary_lost: true, summary: "Ada lives in Lyon." })]} />
    );

    expect(
      screen.getByText(
        "Earlier messages could not be summarized and were dropped from what the model sees."
      )
    ).toBeTruthy();
    expect(screen.queryByText(/were summarized to/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Show the summary the model keeps" }));
    expect(screen.getByText("Ada lives in Lyon.")).toBeTruthy();
  });

  it("offers no toggle when there is no summary text", () => {
    render(<CompactionMarker events={[done({ summary_lost: true, summary: "" })]} />);

    expect(screen.queryByRole("button")).toBeNull();
  });

  it("renders nothing without a compaction", () => {
    const { container } = render(<CompactionMarker events={[]} />);

    expect(container.innerHTML).toBe("");
  });

  it("links each toggle to its own scrollable summary panel", () => {
    render(
      <>
        <CompactionMarker events={[done({ summary: "First." }), done({ summary: "Second." })]} />
        <CompactionMarker events={[done({ summary: "Other turn." })]} />
      </>
    );

    const toggles = screen.getAllByRole("button", { name: "Show summary" });
    const ids = toggles.map((b) => b.getAttribute("aria-controls"));
    expect(ids.every(Boolean)).toBe(true);
    expect(new Set(ids).size).toBe(3);

    fireEvent.click(toggles[1]);
    const panel = document.getElementById(ids[1]);
    expect(panel.textContent).toBe("Second.");
    expect(panel.className).toContain("overflow-y-auto");
    expect(panel.className).toMatch(/max-h-/);
  });
});
