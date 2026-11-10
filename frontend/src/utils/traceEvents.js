// What an assistant turn's trace holds, sorted by where it renders.
//
// The trace (live while streaming, persisted with the message) carries the
// reasoning strip's events -- thinking / tool_call / tool_result, plus a
// leading {"t":"truncated"} marker on a capped replay -- and the compaction
// markers (#614): one {"t":"compaction","phase":"done",...} per compaction the
// turn ran, rendered as separators above the bubble, never inside the strip.
// A /compact run (#585) saves its own `done` event, `manual: true`, on the
// last answer it found: that separator renders below the bubble.
// A failed turn also holds {"t":"error_detail","text","request_id"} (#615): the
// cleaned cause, shown under the red headline. A curated turn holds
// {"t":"curated_error","kind","params"} (#690): what its sentence is
// translated from (utils/curatedTurns.js).

const STRIP_KINDS = new Set(["thinking", "tool_call", "tool_result"]);

const isEvent = (e) => !!e && typeof e === "object";

const isCompactionDone = (e) => isEvent(e) && e.t === "compaction" && e.phase === "done";

/** Whether the trace holds anything the reasoning strip renders. */
export function hasStripEvents(trace) {
  return Array.isArray(trace) && trace.some((e) => isEvent(e) && STRIP_KINDS.has(e.t));
}

const compactionMarkers = (trace) => (Array.isArray(trace) ? trace.filter(isCompactionDone) : []);

/** The compactions the turn ran before its answer, in order (above the bubble). */
export function automaticCompactions(trace) {
  return compactionMarkers(trace).filter((e) => e.manual !== true);
}

/** The /compact runs saved on this answer, in order (below the bubble). */
export function manualCompactions(trace) {
  return compactionMarkers(trace).filter((e) => e.manual === true);
}

const isErrorDetail = (e) =>
  isEvent(e) && e.t === "error_detail" && typeof e.text === "string" && e.text !== "";

/** The cleaned cause of an unexpectedly failed turn (#615):
 * {"t":"error_detail","text","request_id"}, or null. */
export function errorDetailOf(trace) {
  return (Array.isArray(trace) && trace.find(isErrorDetail)) || null;
}

const isCuratedError = (e) => isEvent(e) && e.t === "curated_error";

/** What an error turn keeps of its trace: its compactions (checkpointed before
 * the model call failed), the cause of its failure and the kind of a curated
 * one, or nothing. */
export function errorTurnTrace(trace) {
  if (!Array.isArray(trace)) {
    return null;
  }
  const kept = trace.filter((e) => isCompactionDone(e) || isErrorDetail(e) || isCuratedError(e));
  return kept.length > 0 ? kept : null;
}
