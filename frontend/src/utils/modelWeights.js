// Orphan-model helpers shared by the landing page and the model cards
// (#225/#208).
//
// A KB assistant does not own model files: its `link` is a copy of its base
// model's, so the base's weights are what actually answer. When the base is
// deleted with `?orphan_dependents=true` the assistant survives with
// `weights_available === false` until it is re-bound to another base.

// True when the local model row is a KB assistant (it carries a kb_id, or the
// backend flags it explicitly). Plain installed models return false.
export function isKbAssistant(model) {
  if (!model) {
    return false;
  }
  return model.is_attached_to_kb === true || (model.kb_id !== undefined && model.kb_id !== null);
}

// True ONLY on an explicit `weights_available === false` (an orphan). Unknown
// (undefined/null — remote rows or an older backend) never marks a model as
// missing, so nothing is wrongly blocked.
export function hasMissingWeights(model) {
  return Boolean(model) && model.weights_available === false;
}

// The installed models a conversation can run on: an orphaned KB assistant
// (weights missing) is never offered. Shared by the Model popup and the
// header's model picker (#704).
export function offeredModels(models) {
  return models.filter((model) => !hasMissingWeights(model));
}

// Name of the installed base model whose weights an assistant uses, derived by
// matching `link` against the local list (the assistant's link is a copy of
// its base's). Returns null when the base is gone (deleted -> orphan).
export function findBaseModelName(assistant, localModels) {
  if (!assistant || !Array.isArray(localModels)) {
    return null;
  }
  const base = localModels.find(
    (m) => m.id !== assistant.id && !isKbAssistant(m) && m.link && m.link === assistant.link
  );
  return base ? base.name : null;
}
