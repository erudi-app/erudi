// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, afterEach } from "vitest";
import { render, cleanup, screen } from "@testing-library/react";

import WelcomeModal from "./WelcomeModal.jsx";

// The hardware panel shows the backend's score + label ("53% Fair") with a
// caption below. The caption must be derived from the SAME thresholds the
// backend label uses (hardware/services.py _get_label: 80+ Excellent, 60+
// Good, 40+ Fair, 20+ Poor, else Weak) — a Fair score captioned "Great
// Performance!" is mixed messaging (#303).

const renderWith = (score, label) =>
  render(
    <WelcomeModal
      isOpen
      onClose={() => {}}
      loading={false}
      hardwareInfo={{ global_inference_score: score, global_inference_label: label }}
    />
  );

afterEach(() => {
  cleanup();
});

describe("WelcomeModal hardware caption matches the score tier (#303)", () => {
  it("does not oversell a Fair score", () => {
    renderWith(53, "Fair");

    expect(screen.queryByText(/Great Performance/)).toBeNull();
    expect(screen.queryByText(/Excellent Hardware/)).toBeNull();
    expect(screen.getByText(/Fair Performance/)).toBeTruthy();
  });

  it("captions an Excellent score as excellent", () => {
    renderWith(85, "Excellent");

    expect(screen.getByText(/Excellent Hardware/)).toBeTruthy();
  });

  it("captions a Good score as good, not excellent", () => {
    renderWith(60, "Good");

    expect(screen.queryByText(/Excellent Hardware/)).toBeNull();
    expect(screen.getByText(/Good Performance/)).toBeTruthy();
  });

  it("points Poor and Weak scores to the smaller models", () => {
    renderWith(25, "Poor");
    expect(screen.getByText(/Optimized for Smaller Models/)).toBeTruthy();
    cleanup();

    renderWith(10, "Weak");
    expect(screen.getByText(/Optimized for Smaller Models/)).toBeTruthy();
  });
});

// The machine readout on the page behind this dialog states the size window the
// benchmark produced. The dialog used to carry its own hardcoded range, so on a
// machine whose window was 8-24B the first screen said 8-24B and 1B-12B at once
// (#523).
describe("WelcomeModal size window (#523)", () => {
  const renderWithWindow = (score, min, max) =>
    render(
      <WelcomeModal
        isOpen
        onClose={() => {}}
        loading={false}
        hardwareInfo={{
          global_inference_score: score,
          global_inference_label: "Excellent",
          recommended_param_min: min,
          recommended_param_max: max,
        }}
      />
    );

  it("states the window the benchmark produced, not a fixed one", () => {
    renderWithWindow(85, 8, 24);

    expect(screen.getByText(/from 8B to 24B/)).toBeTruthy();
  });

  it("never contradicts that window with a range of its own", () => {
    renderWithWindow(85, 8, 24);

    // The tier copy is qualitative: no parameter count may appear outside the
    // window sentence, or the two can disagree again.
    expect(screen.queryByText(/1B to 12B/)).toBeNull();
    expect(screen.queryByText(/Mistral 8B|Gemma 12B/)).toBeNull();
  });

  it("says nothing about a window when profiling produced none", () => {
    renderWith(85, "Excellent");

    expect(screen.queryByText(/from .*B to .*B/)).toBeNull();
    // The tier caption still shows: the dialog degrades, it does not go blank.
    expect(screen.getByText(/Excellent Hardware/)).toBeTruthy();
  });
});
