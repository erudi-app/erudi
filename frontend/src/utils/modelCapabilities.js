// Model capability helpers shared by the chat composer.
//
// Whether the image-attach affordance should be enabled for a model. Permissive
// by design: attaching is disabled ONLY when the backend reports an explicit
// `supports_vision === false`, so a model whose capability is unknown (null,
// e.g. a fresh boot before detection) is never wrongly blocked (#133).
export function canAttachImages(model) {
  return model?.supports_vision !== false;
}

// How many images the composer allows per turn for a given model. More capable
// models have more context to spend on image tokens, so scale the cap with the
// model's parameter size. This is a proxy: the backend does not expose a
// context-window field, and param_size is the closest available signal. Unknown
// size falls back to a sensible default. Callers only reach this for
// vision-capable models (the attach affordance is hidden otherwise).
export function maxImagesForModel(model) {
  const size = model?.param_size;
  if (typeof size !== "number" || size <= 0) return 4; // unknown -> default
  if (size < 3) return 2;
  if (size < 8) return 4;
  return 6;
}

// Whether a model can read image input, for the capability badge on model cards.
// Unlike `canAttachImages` (permissive: everything except an explicit false),
// this is a POSITIVE signal — true only when we can affirm vision support — so a
// card shows the image icon only for models we know are multimodal.
//   - `supports_vision === true`: an installed model the engine detected as a VLM.
//   - `category === "vision"`: the pre-download signal — the catalog/HF-search
//     buckets multimodal models into "Vision & Multimodal" from their pipeline
//     tag / name, so it's available before a model is downloaded (when
//     `supports_vision` is still null).
export function modelSupportsVision(model) {
  if (!model) return false;
  if (model.supports_vision === true) return true;
  return model.category === "vision";
}

// Below this many billions of parameters, tool calling, knowledge-base search
// and multi-step reasoning are unreliable (#381, #129): the tools and prompts
// reach the model, it just does not act on them. The QA guidance uses ~4B as
// the floor for anything involving tools, retrieval or reasoning.
export const SMALL_MODEL_PARAM_THRESHOLD_B = 4;

// Parameter count in billions from either signal the frontend carries: the
// backend's numeric `param_size` (billions, null when unmeasured) or the
// metadata string `parameters` ("7B", "1.5B", "270M", "Unknown"). Anything that
// is not a positive size parses to null — unknown must never be mistaken for
// small.
export function parseParamSizeB(value) {
  if (typeof value === "number") {
    return Number.isFinite(value) && value > 0 ? value : null;
  }
  if (typeof value !== "string") return null;
  const match = /^\s*(\d+(?:\.\d+)?)\s*([bm])\s*$/i.exec(value);
  if (!match) return null;
  const amount = parseFloat(match[1]);
  if (!(amount > 0)) return null;
  return match[2].toLowerCase() === "m" ? amount / 1000 : amount;
}

// Whether a model is small enough that the card should warn about tools, KB
// search and reasoning. The numeric `param_size` wins when present (it is the
// measured value); the `parameters` string is the fallback for installed rows
// that only carry metadata. Unknown size -> false (positive signal only).
export function isVerySmallModel(model) {
  if (!model) return false;
  const size = parseParamSizeB(model.param_size) ?? parseParamSizeB(model.parameters);
  return size !== null && size < SMALL_MODEL_PARAM_THRESHOLD_B;
}

// Whether web search is unavailable for a model because it cannot execute
// tools at all (#570). The `web_search` tool is silently dropped server-side
// (backend/src/agents/kb_mode.py) whenever `supports_tools` or
// `supports_tools_wire` is not truthy, so a conversation's header toggle can
// sit ON with zero effect and zero feedback. This is a POSITIVE-knowledge
// gate, mirroring `canAttachImages`'s vision-gating philosophy but inverted:
// the toggle is disabled ONLY when a flag explicitly says `false`. Unknown
// (null/undefined, e.g. detection not run yet) or a missing/orphaned model
// never disables it -- only a proven incapacity does.
export function webSearchUnavailableForModel(model) {
  if (!model) return false;
  return model.supports_tools === false || model.supports_tools_wire === false;
}
