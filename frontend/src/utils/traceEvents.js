// What an assistant turn's trace holds, sorted by where it renders.
//
// The trace (live while streaming, persisted with the message) carries the
// reasoning strip's events -- thinking / tool_call / tool_result, plus a
// leading {"t":"truncated"} marker on a capped replay -- and the compaction
// markers (#614): one {"t":"compaction","phase":"done",...} per compaction the
// turn ran, rendered as separators above the bubble, never inside the strip.
// A /compact run (#585) saves its own `done` event, `manual: true`, on the
// last answer it found: that separator renders below the bubble.

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

/** What an error turn keeps of its trace: its compactions (the compaction was
 * checkpointed before the model call failed), or nothing. */
export function compactionsOnly(trace) {
  const kept = compactionMarkers(trace);
  return kept.length > 0 ? kept : null;
}
