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
    expect(out).toBe("$$a$$\n\n```\n\\(b\\)\n```\n\n$$c$$");
  });
});

describe("display math inside lists and quotes (#690)", () => {
  it("closes a display block inside a list item and keeps the rest of the answer", () => {
    const { container } = renderMarkdown(
      "1. Step:\n   \\[\n   a = b\n   \\]\n2. Next \\(c\\)\n\nAfter \\(d\\)"
    );
    expect(container.querySelectorAll(".katex-display").length).toBe(1);
    expect(container.querySelectorAll("li").length).toBe(2);
    expect(container.querySelectorAll(".katex").length).toBe(3);
    expect(container.textContent).toContain("After");
    expect(container.textContent).not.toContain("$");
  });

  it("closes a display block inside a blockquote", () => {
    const { container } = renderMarkdown("> Note:\n> \\[\n> a = b\n> \\]\n\nAfter \\(d\\)");
    expect(container.querySelector("blockquote .katex-display")).toBeTruthy();
    expect(container.textContent).not.toContain("$");
    expect(container.querySelectorAll(".katex").length).toBe(2);
  });
});

describe("inline conversion emits double dollars (#690)", () => {
  it("renders \\[x\\] after a digit mid-line", () => {
    const { container } = renderMarkdown("Let 2 \\[x\\] be given.");
    expect(container.querySelector(".katex")).toBeTruthy();
    expect(container.textContent).not.toContain("$");
  });

  it("lets a lone currency dollar before converted math stay literal", () => {
    const { container } = renderMarkdown("Cost 212,75 $ and \\(3x\\) here.");
    expect(container.querySelectorAll(".katex").length).toBe(1);
    expect(container.textContent).toContain("212,75 $ and");
  });

  it("renders a converted span alone on its line", () => {
    const { container } = renderMarkdown("\\(x+1\\)\n\nnext");
    expect(container.querySelector(".katex")).toBeTruthy();
    expect(container.textContent).not.toContain("$");
  });
});

describe("what is not converted (#690)", () => {
  it("stops an inline code span at a blank line", () => {
    const { text } = convertTexDelimiters("a ` b\n\n\\(x\\) c ` d");
    expect(text).toBe("a ` b\n\n$$x$$ c ` d");
  });

  it("leaves a 4-space indented code block alone", () => {
    const { container } = renderMarkdown("Code:\n\n    \\(x\\) and \\[y\\]\n\nthen \\(z\\)");
    expect(container.querySelector("pre code").textContent).toContain("\\(x\\) and \\[y\\]");
    expect(container.querySelectorAll(".katex").length).toBe(1);
  });

  it("converts the paragraph of a list item indented by four spaces", () => {
    const { text } = convertTexDelimiters("- item\n\n    more \\(x\\)");
    expect(text).toBe("- item\n\n    more $$x$$");
  });

  it("leaves a link destination alone", () => {
    const { text } = convertTexDelimiters("[a](http://x.test/\\(1\\)) and \\(y\\)");
    expect(text).toBe("[a](http://x.test/\\(1\\)) and $$y$$");
  });

  it("leaves \\[text\\](url) alone", () => {
    const { text } = convertTexDelimiters("see \\[text\\](http://x.test)");
    expect(text).toBe("see \\[text\\](http://x.test)");
  });

  it("leaves a citation such as \\[1\\] and \\[1, 2\\] as text", () => {
    const { container } = renderMarkdown("As shown \\[1\\] and \\[1, 2\\].");
    expect(container.querySelector(".katex")).toBeNull();
    expect(container.textContent).toContain("[1]");
    expect(convertTexDelimiters("As \\[1\\]").text).toBe("As \\[1\\]");
  });
});
