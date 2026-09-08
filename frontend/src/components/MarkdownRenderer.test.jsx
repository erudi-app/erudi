// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";

import MarkdownRenderer from "./MarkdownRenderer.jsx";

// Local models (Qwen3 in particular) routinely emit inline TeX between single
// dollars — "$ 3 - 1 = 2 $", "$ \frac{14}{2} = 7 $" — so single-dollar math
// must render (#303). Single-dollar parsing is a known currency footgun:
// remark-math's default happily turns "I have $5 and $10" into math("5 and ").
// The DELIBERATE configuration here: single-dollar math stays enabled, but a
// span whose opening "$" is immediately followed by a digit is treated as
// currency and kept as literal text. Trade-off (documented): math that starts
// with a bare digit right after the dollar ("$3x+1$") stays literal — models
// pad their math ("$ 3x+1 $") or open with a symbol, both of which render.

const renderMarkdown = (content) => render(<MarkdownRenderer content={content} />);

afterEach(() => {
  cleanup();
});

describe("MarkdownRenderer math support (#303)", () => {
  it("renders single-dollar inline math as a katex element", () => {
    const { container } = renderMarkdown("The square is $x^2$ here.");

    expect(container.querySelector(".katex")).toBeTruthy();
    // The raw TeX source must no longer be visible.
    expect(container.textContent).not.toContain("$x^2$");
  });

  it("renders Qwen-style space-padded math", () => {
    const { container } = renderMarkdown("So $ \\frac{14}{2} = 7 $ apples.");

    expect(container.querySelector(".katex")).toBeTruthy();
    // The dollar-delimited raw source is gone (KaTeX keeps the TeX source in a
    // visually-hidden MathML annotation, so match the delimiters, not the TeX).
    expect(container.textContent).not.toContain("$ \\frac{14}{2} = 7 $");
  });

  it("renders double-dollar display math", () => {
    const { container } = renderMarkdown("$$\nE = mc^2\n$$");

    expect(container.querySelector(".katex")).toBeTruthy();
  });

  it("does not mangle dollar amounts", () => {
    const { container } = renderMarkdown("I have $5 and $10 in my pocket.");

    expect(container.querySelector(".katex")).toBeNull();
    expect(container.textContent).toContain("$5 and $10");
  });

  it("does not mangle thousand-separated prices or ranges", () => {
    const { container } = renderMarkdown("It costs $20,000 and $30,000, or $5-$10 per unit.");

    expect(container.querySelector(".katex")).toBeNull();
    expect(container.textContent).toContain("$20,000 and $30,000");
    expect(container.textContent).toContain("$5-$10");
  });

  it("keeps plain markdown rendering intact", () => {
    const { container } = renderMarkdown("Some **bold** text and `inline code`.");

    expect(container.querySelector("strong").textContent).toBe("bold");
    expect(container.querySelector("code").textContent).toBe("inline code");
  });

  // French/European currency puts the amount before the sign with a space
  // ("212,75 $"), which the digit-after-"$" rule alone does not catch — the
  // character right after the opening "$" is a space, not a digit, so
  // remark-math pairs the two signs and everything between them renders as
  // broken inline math (#501).
  it("does not mangle French currency written before the dollar sign", () => {
    const { container } = renderMarkdown(
      "Pour une PME, le **cout d'acquisition client** se situe autour de **212,75 $ par mois par poste** " +
        "*(source interne)*, tandis que le forfait annuel de 1 500 $ s'applique aux equipes plus larges."
    );

    expect(container.querySelector(".katex")).toBeNull();
    expect(container.querySelector("strong")).toBeTruthy();
    expect(container.textContent).toContain("212,75 $");
    expect(container.textContent).toContain("1 500 $");
  });

  it("guards currency per-span, leaving a real formula on the same line rendering as katex", () => {
    const { container } = renderMarkdown(
      "Le prix est 12 $ et 15 $ mais la formule $x^2 + 1$ est correcte."
    );

    expect(container.querySelector(".katex")).toBeTruthy();
    expect(container.textContent).toContain("12 $ et 15 $");
  });

  // French typography puts a no-break space before the currency sign, and
  // models emit it: the ordinary no-break space U+00A0 ("\u00A0") and the
  // narrow no-break space U+202F ("\u202F"). Neither is an ASCII " ", so both
  // need their own case alongside the plain-space rule above (#501).
  it("does not mangle French currency separated from the sign by a no-break space (U+00A0)", () => {
    const { container } = renderMarkdown(
      "Pour une PME, le **cout d'acquisition client** se situe autour de **212,75\u00A0$ par mois par poste** " +
        "*(source interne)*, tandis que le forfait annuel de 1\u00A0500\u00A0$ s'applique aux equipes plus larges."
    );

    expect(container.querySelector(".katex")).toBeNull();
    expect(container.querySelector("strong")).toBeTruthy();
    expect(container.textContent).toContain("212,75\u00A0$");
    expect(container.textContent).toContain("500\u00A0$");
  });

  it("does not mangle French currency separated from the sign by a narrow no-break space (U+202F)", () => {
    const { container } = renderMarkdown(
      "Pour une PME, le **cout d'acquisition client** se situe autour de **212,75\u202F$ par mois par poste** " +
        "*(source interne)*, tandis que le forfait annuel de 1\u202F500\u202F$ s'applique aux equipes plus larges."
    );

    expect(container.querySelector(".katex")).toBeNull();
    expect(container.querySelector("strong")).toBeTruthy();
    expect(container.textContent).toContain("212,75\u202F$");
    expect(container.textContent).toContain("500\u202F$");
  });
});
