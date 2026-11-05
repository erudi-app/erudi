// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import i18n from "../i18n";

// The weights missing badge sits beside the model name. A long name must be
// cut with an ellipsis and stay readable in full on hover, and the longer
// French label must never cover the title.

vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: vi.fn() }),
}));

import ModelCard from "./ModelCard.jsx";

const LONG_NAME =
  "Qwen2.5-Coder-32B-Instruct-Abliterated-Distilled-Reasoning-Edition-GGUF-Q4_K_M-Extra-Long";

const missingModel = { id: 3, name: LONG_NAME, weights_available: false };

afterEach(async () => {
  cleanup();
  await i18n.changeLanguage("en");
});

describe("weights missing badge beside a long model name", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en");
  });

  it("cuts the title with an ellipsis and exposes the full name on hover", () => {
    render(<ModelCard model={missingModel} type="local" />);
    const title = screen.getByRole("heading", { level: 3 });
    expect(title.textContent).toBe(LONG_NAME);
    expect(title.getAttribute("title")).toBe(LONG_NAME);
    expect(title.className).toContain("truncate");
    expect(title.className).toContain("min-w-0");
  });

  it("keeps the badge as a non-shrinking sibling of the title block", () => {
    render(<ModelCard model={missingModel} type="local" />);
    const title = screen.getByRole("heading", { level: 3 });
    const badge = screen.getByText("Weights missing");
    expect(badge.className).toContain("flex-shrink-0");
    // The title sits in its own block; the badge is a sibling of that block.
    expect(badge.parentElement).toBe(title.parentElement.parentElement);
  });

  it("gives the French label the same non-shrinking, sibling placement", async () => {
    await i18n.changeLanguage("fr");
    render(<ModelCard model={missingModel} type="local" />);
    const title = screen.getByRole("heading", { level: 3 });
    const badge = screen.getByText("Poids manquants");
    expect(badge.className).toContain("flex-shrink-0");
    expect(badge.parentElement).toBe(title.parentElement.parentElement);
    expect(title.className).toContain("truncate");
  });
});
