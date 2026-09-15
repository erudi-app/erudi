// Per-model sampling defaults (#388).
//
// The backend resolves, per catalog row, the sampling a fresh conversation or
// arena panel should start from (`sampling_defaults`: what the publisher ships
// in the base repo's generation_config.json, else the quant repo's, else what
// the base model card recommends in prose — the winning stage is `source`).
// The UI seeds its sliders from that block and never hard-codes a per-model
// value itself.
//
// `max_tokens` / `max_tokens_cap` are deliberately NOT read here. How long an
// answer may get is not a setting any more: the backend computes it per model
// call from the context window the model is running in (see
// docs/guides/conversations.md, "Output budget"), so there is no field to seed
// and no ceiling to enforce in the interface.
//
// The fallback mirrors the backend's constants (src/database/generation_hints.py):
// 0.2 / 0.95, validated by the #129 eval campaign.
export const FALLBACK_SAMPLING = Object.freeze({
  temperature: 0.2,
  topP: 0.95,
});

// `sampling_defaults.source` when the publisher gives no usable sampling
// recommendation and the neutral constants above apply.
export const SAMPLING_SOURCE_NONE = "none";

const numberOr = (value, fallback) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

/**
 * The sampling a panel should start from for `model` (a `/llms/local` row or
 * null), in the UI's camelCase shape. Always a fresh object; falls back per
 * key so a partial block never yields NaN sliders.
 */
export function defaultsFor(model) {
  const block = model?.sampling_defaults;
  if (!block || typeof block !== "object") {
    return { ...FALLBACK_SAMPLING };
  }
  return {
    temperature: numberOr(block.temperature, FALLBACK_SAMPLING.temperature),
    topP: numberOr(block.top_p, FALLBACK_SAMPLING.topP),
  };
}

/**
 * True when the backend resolved `model`'s sampling from nothing: the
 * publisher ships no generation_config values and its model card carries no
 * recommendation. Unknown (no block at all — a Hugging Face search result, a
 * model not selected yet) is NOT "none": the note is only shown on a verdict.
 */
export function hasNoPublisherRecommendation(model) {
  return model?.sampling_defaults?.source === SAMPLING_SOURCE_NONE;
}
