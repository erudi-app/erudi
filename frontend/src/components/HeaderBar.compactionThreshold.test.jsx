// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act } from "@testing-library/react";

// #607 — per-conversation Compaction threshold control in the settings panel,
// beside the Reasoning effort select. Hidden by default so the Arena (which
// shares HeaderBar but never compacts) keeps an unchanged panel;
// ConversationPage opts in with showCompactionThreshold.

import HeaderBar from "./HeaderBar.jsx";
import i18n from "../i18n";

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver || ResizeObserverStub;

const renderBar = (props = {}) =>
  render(<HeaderBar onApply={() => {}} onCustomizePrompt={() => {}} {...props} />);

const openSettings = () => fireEvent.click(screen.getByLabelText("Toggle settings"));
const thresholdSelect = () => screen.findByRole("combobox", { name: "Compaction threshold" });

afterEach(async () => {
  cleanup();
  await i18n.changeLanguage("en");
});

describe("HeaderBar compaction threshold control (#607)", () => {
  it("is hidden by default (the Arena never compacts)", async () => {
    renderBar();
    openSettings();
    await screen.findAllByRole("slider");
    expect(screen.queryByRole("combobox", { name: "Compaction threshold" })).toBeNull();
  });

  it("offers 50 % to 90 % in steps of 5", async () => {
    renderBar({ showCompactionThreshold: true, initialCompactionThreshold: 80 });
    openSettings();
    const select = await thresholdSelect();
    const options = Array.from(select.querySelectorAll("option"));
    expect(options.map((o) => o.value)).toEqual([
      "50",
      "55",
      "60",
      "65",
      "70",
      "75",
      "80",
      "85",
      "90",
    ]);
    expect(options[0].textContent).toBe("50%");
  });

  it("formats the percentages for the active language", async () => {
    renderBar({ showCompactionThreshold: true, initialCompactionThreshold: 80 });
    openSettings();
    await act(() => i18n.changeLanguage("fr"));
    const select = await screen.findByRole("combobox", { name: "Seuil de compaction" });
    const label = select.querySelector('option[value="80"]').textContent;
    // French puts a (narrow) no-break space before the sign: "80 %".
    expect(label.replace(/\s/g, " ")).toBe("80 %");
  });

  it("reflects initialCompactionThreshold", async () => {
    renderBar({ showCompactionThreshold: true, initialCompactionThreshold: 65 });
    openSettings();
    expect((await thresholdSelect()).value).toBe("65");
  });

  it("reports the new value as a number through onCompactionThresholdChange", async () => {
    const onCompactionThresholdChange = vi.fn();
    renderBar({
      showCompactionThreshold: true,
      initialCompactionThreshold: 80,
      onCompactionThresholdChange,
    });
    openSettings();
    const select = await thresholdSelect();
    fireEvent.change(select, { target: { value: "55" } });
    expect(onCompactionThresholdChange).toHaveBeenCalledWith(55);
    expect(select.value).toBe("55");
  });

  it("syncs with a later initialCompactionThreshold prop change (hydration)", async () => {
    const { rerender } = renderBar({
      showCompactionThreshold: true,
      initialCompactionThreshold: 80,
    });
    openSettings();
    const select = await thresholdSelect();
    expect(select.value).toBe("80");
    rerender(
      <HeaderBar
        onApply={() => {}}
        onCustomizePrompt={() => {}}
        showCompactionThreshold
        initialCompactionThreshold={50}
      />
    );
    expect(select.value).toBe("50");
  });
});
