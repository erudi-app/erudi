// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import i18n from "../../i18n";
import { formatNumber } from "../../i18n/format";

import ModelInfoModal from "./ModelInfoModal.jsx";

// Same rule as the card (#387): the details modal shows the size through the
// locale formatter, measured bytes first, metadata string next.

const open = (modelInfo) =>
  render(<ModelInfoModal modelInfo={modelInfo} isOpen onClose={vi.fn()} onDownload={vi.fn()} />);

beforeEach(async () => {
  await i18n.changeLanguage("fr");
});
afterEach(async () => {
  cleanup();
  await i18n.changeLanguage("en");
});

describe("ModelInfoModal size in French (#387)", () => {
  it("formats the metadata size string in the active locale", () => {
    open({ name: "Gemma 1B", size: "~0.7 GB", parameters: "1B" });
    expect(screen.getByText("~0,7 Go")).toBeTruthy();
  });

  it("prefers the measured artifact size", () => {
    open({
      name: "Gemma 1B",
      size: "~0.7 GB",
      parameters: "1B",
      artifact_size_bytes: 4_700_000_000,
    });
    expect(screen.getByText("4,7 Go")).toBeTruthy();
    expect(screen.queryByText(/0\.7/)).toBeNull();
  });

  it("keeps the raw value when nothing can be formatted", () => {
    open({ name: "Bare", size: "Inconnu", parameters: "Inconnu" });
    expect(screen.getAllByText("Inconnu").length).toBeGreaterThan(0);
  });
});

describe("ModelInfoModal numbers in French (#765)", () => {
  it("groups the context window digits with the language's separator", () => {
    open({
      name: "Gemma 1B",
      size: "~0.7 GB",
      parameters: "1B",
      context_window: 40960,
      allocated_context_window: 8192,
    });
    // Exact text, no whitespace normalization: the French separator is a
    // narrow no-break space.
    const exact = (text) => screen.getByText((_, node) => node.textContent === text);
    expect(exact(`${formatNumber(40960)} tokens`)).toBeTruthy();
    expect(exact(`${formatNumber(8192)} tokens`)).toBeTruthy();
    expect(screen.queryByText("40960 tokens")).toBeNull();
  });

  it("groups downloads and likes and localizes the parameter count", () => {
    open({
      name: "Gemma 1B",
      size: "~0.7 GB",
      parameters: "4.0B",
      downloads: "1234567",
      likes: 8901,
    });
    const exact = (text) => screen.getByText((_, node) => node.textContent === text);
    expect(exact(formatNumber(1234567))).toBeTruthy();
    expect(exact(formatNumber(8901))).toBeTruthy();
    expect(exact("4,0B")).toBeTruthy();
  });
});
