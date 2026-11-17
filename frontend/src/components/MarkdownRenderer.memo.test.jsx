// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";

// #618 — MarkdownRenderer is memoized on its text: an unchanged message is not
// parsed again (the Arena panels re-render every panel on each streamed
// chunk), and a message whose text changes is.

const { parses } = vi.hoisted(() => ({ parses: [] }));

vi.mock("react-markdown", () => ({
  default: ({ children, components }) => {
    parses.push({ text: String(children), components });
    return <div data-testid="md">{children}</div>;
  },
}));
vi.mock("katex/dist/katex.min.css", () => ({}));

import MarkdownRenderer from "./MarkdownRenderer.jsx";

afterEach(() => {
  cleanup();
  parses.length = 0;
});

describe("MarkdownRenderer memoization", () => {
  it("does not parse again when its parent re-renders with the same text", () => {
    const { rerender } = render(<MarkdownRenderer content="same **text**" />);
    rerender(<MarkdownRenderer content="same **text**" />);
    rerender(<MarkdownRenderer content="same **text**" />);
    expect(parses).toHaveLength(1);
  });

  it("parses again, and shows the new text, when the text changes", () => {
    const { rerender } = render(<MarkdownRenderer content="before" />);
    rerender(<MarkdownRenderer content="after" />);
    expect(parses.map((p) => p.text)).toEqual(["before", "after"]);
    expect(screen.getByTestId("md").textContent).toBe("after");
  });

  it("hands react-markdown the same component map on every parse", () => {
    const { rerender } = render(<MarkdownRenderer content="one" />);
    rerender(<MarkdownRenderer content="two" />);
    expect(parses[0].components).toBe(parses[1].components);
  });
});
