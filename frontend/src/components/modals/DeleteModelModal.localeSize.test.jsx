// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, cleanup } from "@testing-library/react";
import i18n from "../../i18n";

import DeleteModelModal from "./DeleteModelModal.jsx";

// The size the delete dialog announces follows the interface language and the
// card's convention (decimal GB, "Go" in French), not the backend's English
// "~0.7 GB" string.

// The size line belongs to the guarded delete of a base with dependents.
const dependents = {
  assistants: [{ id: 2, name: "Helper" }],
  own_conversation_count: 1,
  total_conversation_count: 1,
};

const open = (model) =>
  render(
    <DeleteModelModal
      isOpen
      model={model}
      dependents={dependents}
      onConfirm={vi.fn()}
      onCancel={vi.fn()}
    />
  );

// The sentence sits in a <p> beside the Trans-rendered question.
const freesLine = () => document.body.textContent.match(/Sa suppression libère[^.]*\./)[0];

beforeEach(async () => {
  await i18n.changeLanguage("fr");
});
afterEach(async () => {
  cleanup();
  await i18n.changeLanguage("en");
});

describe("DeleteModelModal size in French", () => {
  it("formats the metadata size string in the interface language", () => {
    open({ name: "Gemma 1B", size: "~0.7 GB" });
    expect(freesLine()).toContain("0,7 Go");
    expect(freesLine()).not.toContain("0.7");
    expect(freesLine()).not.toContain("GB");
  });

  it("prefers the measured size", () => {
    open({ name: "Gemma 1B", size: "~0.7 GB", artifact_size_bytes: 2_279_000_000 });
    expect(freesLine()).toContain("2,3 Go");
  });

  it("says nothing about the size when it is unknown", () => {
    open({ name: "Gemma 1B", size: "Unknown" });
    expect(document.body.textContent).not.toContain("libère");
  });
});
