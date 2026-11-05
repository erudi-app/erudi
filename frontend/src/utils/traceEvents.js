// What an assistant turn's trace holds, sorted by where it renders.
//
// The trace (live while streaming, persisted with the message) carries the
// reasoning strip's events -- thinking / tool_call / tool_result, plus a
// leading {"t":"truncated"} marker on a capped replay -- and the compaction
// markers (#614): one {"t":"compaction","phase":"done",...} per compaction the
// turn ran, rendered as separators above the bubble, never inside the strip.
// A failed turn also holds {"t":"error_detail","text","request_id"} (#615): the
// cleaned cause, shown under the red headline.

const STRIP_KINDS = new Set(["thinking", "tool_call", "tool_result"]);

const isEvent = (e) => !!e && typeof e === "object";

const isCompactionDone = (e) => isEvent(e) && e.t === "compaction" && e.phase === "done";

/** Whether the trace holds anything the reasoning strip renders. */
export function hasStripEvents(trace) {
  return Array.isArray(trace) && trace.some((e) => isEvent(e) && STRIP_KINDS.has(e.t));
}

/** The turn's compaction markers, in order. */
export function compactionMarkers(trace) {
  return Array.isArray(trace) ? trace.filter(isCompactionDone) : [];
}

const isErrorDetail = (e) =>
  isEvent(e) && e.t === "error_detail" && typeof e.text === "string" && e.text !== "";

/** The cleaned cause of an unexpectedly failed turn (#615):
 * {"t":"error_detail","text","request_id"}, or null. */
export function errorDetailOf(trace) {
  return (Array.isArray(trace) && trace.find(isErrorDetail)) || null;
}

/** What an error turn keeps of its trace: its compactions (checkpointed before
 * the model call failed) and the cause of its failure, or nothing. */
export function errorTurnTrace(trace) {
  if (!Array.isArray(trace)) {
    return null;
  }
  const kept = trace.filter((e) => isCompactionDone(e) || isErrorDetail(e));
  return kept.length > 0 ? kept : null;
}
