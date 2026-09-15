import { describe, it, expect } from "vitest";
import {
  FALLBACK_SAMPLING,
  SAMPLING_SOURCE_NONE,
  defaultsFor,
  hasNoPublisherRecommendation,
} from "./samplingDefaults";

// Per-model sampling defaults (#388): the backend resolves them per row
// (`sampling_defaults`: base generation_config > quant generation_config >
// model card, else "none") and the UI seeds its sliders from that block.
// Without one, the fallback is the exact pair the backend uses (0.2 / 0.95).
//
// The block also carries `max_tokens` / `max_tokens_cap`, which this helper
// deliberately drops: the output budget is computed by the backend from the
// running context window, not chosen in the interface.

const QWEN3 = {
  id: 1,
  name: "Qwen3 0.6B",
  sampling_defaults: {
    temperature: 0.6,
    top_p: 0.95,
    max_tokens: 1024,
    max_tokens_cap: 8192,
    top_k: 20,
    source: "base_generation_config",
  },
};

describe("defaultsFor", () => {
  it("reads the model's resolved block into the UI's camelCase shape", () => {
    expect(defaultsFor(QWEN3)).toEqual({ temperature: 0.6, topP: 0.95 });
  });

  it("ignores the block's output-budget keys", () => {
    // They stay on the wire (the backend still uses them as its fallback), but
    // nothing in the interface reads or shows them.
    const d = defaultsFor(QWEN3);
    expect(d).not.toHaveProperty("maxTokens");
    expect(d).not.toHaveProperty("maxTokensCap");
  });

  it("falls back to the backend constants without a model or a block", () => {
    expect(defaultsFor(null)).toEqual(FALLBACK_SAMPLING);
    expect(defaultsFor(undefined)).toEqual(FALLBACK_SAMPLING);
    expect(defaultsFor({ id: 2, name: "no hints" })).toEqual(FALLBACK_SAMPLING);
    expect(defaultsFor({ sampling_defaults: null })).toEqual(FALLBACK_SAMPLING);
  });

  it("returns a fresh object each time (callers spread it into state)", () => {
    const a = defaultsFor(null);
    a.temperature = 1.9;
    expect(defaultsFor(null).temperature).toBe(FALLBACK_SAMPLING.temperature);
  });

  it("falls back per key when a value is missing or not a number", () => {
    const d = defaultsFor({ sampling_defaults: { temperature: "0.6", top_p: 0.5 } });
    expect(d).toEqual({ temperature: FALLBACK_SAMPLING.temperature, topP: 0.5 });
  });
});

describe("hasNoPublisherRecommendation", () => {
  it("is true only on the backend's explicit 'none' verdict", () => {
    expect(
      hasNoPublisherRecommendation({ sampling_defaults: { source: SAMPLING_SOURCE_NONE } })
    ).toBe(true);
    expect(hasNoPublisherRecommendation({ sampling_defaults: { source: "none" } })).toBe(true);
  });

  it("is false when a recommendation exists, whatever the stage", () => {
    for (const source of ["base_generation_config", "quant_generation_config", "model_card"]) {
      expect(hasNoPublisherRecommendation({ sampling_defaults: { source } })).toBe(false);
    }
  });

  it("is false when nothing is known (no model, no block, no source)", () => {
    expect(hasNoPublisherRecommendation(null)).toBe(false);
    expect(hasNoPublisherRecommendation(undefined)).toBe(false);
    expect(hasNoPublisherRecommendation({ name: "hf search hit" })).toBe(false);
    expect(hasNoPublisherRecommendation({ sampling_defaults: null })).toBe(false);
    expect(hasNoPublisherRecommendation({ sampling_defaults: { temperature: 0.2 } })).toBe(false);
  });
});
