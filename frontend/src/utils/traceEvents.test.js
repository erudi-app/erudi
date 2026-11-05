import { describe, it, expect } from "vitest";
import { compactionMarkers, errorDetailOf, errorTurnTrace, hasStripEvents } from "./traceEvents";

const done = (summarized) => ({
  t: "compaction",
  phase: "done",
  summarized,
  reason: "length",
  summary: "S.",
  summary_lost: false,
});

describe("trace events (#614)", () => {
  it("finds the strip's events among a turn's trace", () => {
    expect(hasStripEvents([{ t: "thinking", text: "hm" }])).toBe(true);
    expect(hasStripEvents([{ t: "tool_call", name: "x" }])).toBe(true);
    expect(hasStripEvents([{ t: "tool_result", text: "4" }])).toBe(true);
    // A trace holding only a compaction (and a truncated marker) has none.
    expect(hasStripEvents([done(3)])).toBe(false);
    expect(hasStripEvents([done(3), { t: "truncated" }])).toBe(false);
    expect(hasStripEvents(null)).toBe(false);
    expect(hasStripEvents(undefined)).toBe(false);
  });

  it("lists the compaction markers in order, done events only", () => {
    const trace = [done(1), { t: "thinking", text: "hm" }, { t: "compaction", phase: "start" }];
    trace.push(done(2));

    expect(compactionMarkers(trace)).toEqual([done(1), done(2)]);
    expect(compactionMarkers(null)).toEqual([]);
    expect(compactionMarkers([null, "x", { t: "answer" }])).toEqual([]);
  });

  it("keeps only the compactions and the cause of an error turn, or nothing", () => {
    const cause = { t: "error_detail", text: "boom", request_id: "be-1" };
    expect(errorTurnTrace([done(1), { t: "thinking", text: "hm" }])).toEqual([done(1)]);
    expect(errorTurnTrace([{ t: "thinking", text: "hm" }])).toBeNull();
    expect(errorTurnTrace([])).toBeNull();
    expect(errorTurnTrace([done(1), { t: "thinking", text: "hm" }, cause])).toEqual([
      done(1),
      cause,
    ]);
    expect(errorTurnTrace([cause])).toEqual([cause]);
  });

  it("reads the cause of a failed turn from its trace (#615)", () => {
    const cause = { t: "error_detail", text: "boom", request_id: "be-1" };
    expect(errorDetailOf([done(1), cause])).toEqual(cause);
    expect(errorDetailOf([done(1)])).toBeNull();
    expect(errorDetailOf([{ t: "error_detail", text: "" }])).toBeNull();
    expect(errorDetailOf(null)).toBeNull();
    // An error_detail never mounts the reasoning strip.
    expect(hasStripEvents([cause])).toBe(false);
  });
});
