// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";

import MarkdownRenderer from "./MarkdownRenderer.jsx";
import { convertTexDelimiters } from "./MarkdownRenderer.jsx";

// Qwen2.5 writes math between \( ... \) and \[ ... \]. Markdown reads "\("
// as an escaped parenthesis before remark-math ever sees it, so the
// delimiters are converted to dollars in the raw text first (#690).

afterEach(cleanup);

const renderMarkdown = (content) => render(<MarkdownRenderer content={content} />);

describe("MarkdownRenderer \\( \\) and \\[ \\] math (#690)", () => {
  it("renders inline \\(3x\\) as math, though a digit follows the dollar", () => {
    const { container } = renderMarkdown("The term \\(3x\\) grows.");
    expect(container.querySelector(".katex")).toBeTruthy();
    expect(container.querySelector(".katex-display")).toBeNull();
    expect(container.textContent).not.toContain("\\(");
    expect(container.textContent).not.toContain("$");
  });

  it("renders a digit followed by a space then \\(x\\) as math", () => {
    const { container } = renderMarkdown("Let 2 \\(x\\) be given.");
    expect(container.querySelector(".katex")).toBeTruthy();
    expect(container.textContent).toContain("Let 2 ");
  });

  it("renders a display \\[ ... \\] as display math", () => {
    const { container } = renderMarkdown("Then:\n\n\\[ E = mc^2 \\]\n\nDone.");
    expect(container.querySelector(".katex-display")).toBeTruthy();
    expect(container.textContent).not.toContain("\\[");
  });

  it("renders a \\[ spanning several lines", () => {
    const { container } = renderMarkdown("\\[\n\\frac{a}{b}\n= c\n\\]");
    expect(container.querySelector(".katex-display")).toBeTruthy();
    expect(container.textContent).not.toContain("\\[");
    expect(container.textContent).not.toContain("\\]");
  });

  it("keeps an unclosed \\( as text while streaming, and renders it once closed", () => {
    const partial = renderMarkdown("So \\(x^2 + ");
    expect(partial.container.querySelector(".katex")).toBeNull();
    expect(partial.container.textContent).toContain("x^2 +");
    partial.unmount();

    const done = renderMarkdown("So \\(x^2 + 1\\) holds.");
    expect(done.container.querySelector(".katex")).toBeTruthy();
  });

  it("keeps an unclosed \\[ as text while streaming", () => {
    const { container } = renderMarkdown("So\n\n\\[ x^2 + ");
    expect(container.querySelector(".katex")).toBeNull();
  });

  it("still keeps currency literal", () => {
    const fr = renderMarkdown("Le cout est de 212,75 $ par mois.");
    expect(fr.container.querySelector(".katex")).toBeNull();
    expect(fr.container.textContent).toContain("212,75 $");
    fr.unmount();

    const us = renderMarkdown("I have $5 and $10 in my pocket.");
    expect(us.container.querySelector(".katex")).toBeNull();
    expect(us.container.textContent).toContain("$5 and $10");
  });

  it("keeps currency literal next to converted math", () => {
    const { container } = renderMarkdown("It costs 12 $ or 15 $, and \\(3x\\) grows.");
    expect(container.querySelectorAll(".katex").length).toBe(1);
    expect(container.textContent).toContain("12 $ or 15 $");

    const after = renderMarkdown("\\(3x\\) grows and it costs 212,75 $.");
    expect(after.container.querySelectorAll(".katex").length).toBe(1);
    expect(after.container.textContent).toContain("212,75 $");
  });

  it("leaves $$ ... $$ and $ ... $ working as before", () => {
    const display = renderMarkdown("$$\nE = mc^2\n$$");
    expect(display.container.querySelector(".katex-display")).toBeTruthy();
    display.unmount();

    const inline = renderMarkdown("The square is $x^2$ here.");
    expect(inline.container.querySelector(".katex")).toBeTruthy();
  });

  it("does not touch the delimiters inside a fenced code block", () => {
    const { container } = renderMarkdown("```\nA \\(x\\) and \\[y\\]\n```");
    expect(container.querySelector(".katex")).toBeNull();
    expect(container.querySelector("pre code").textContent).toContain("A \\(x\\) and \\[y\\]");
  });

  it("does not touch the delimiters inside inline code", () => {
    const { container } = renderMarkdown("Write `\\(x\\)` or `\\[y\\]` in LaTeX.");
    expect(container.querySelector(".katex")).toBeNull();
    const codes = [...container.querySelectorAll("code")].map((c) => c.textContent);
    expect(codes).toEqual(["\\(x\\)", "\\[y\\]"]);
  });

  it("treats an unclosed fence as code to the end", () => {
    const { container } = renderMarkdown("```\nA \\(x\\)");
    expect(container.querySelector(".katex")).toBeNull();
  });

  it("converts around a closed fence, not inside it", () => {
    const text = "\\(a\\)\n\n```\n\\(b\\)\n```\n\n\\(c\\)";
    const { text: out } = convertTexDelimiters(text);
    expect(out).toBe("$a$\n\n```\n\\(b\\)\n```\n\n$c$");
  });
});
