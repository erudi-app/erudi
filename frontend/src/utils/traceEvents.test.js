import { describe, it, expect } from "vitest";
import { compactionMarkers, compactionsOnly, hasStripEvents } from "./traceEvents";

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

  it("keeps only the compactions of an error turn, or nothing", () => {
    expect(compactionsOnly([done(1), { t: "thinking", text: "hm" }])).toEqual([done(1)]);
    expect(compactionsOnly([{ t: "thinking", text: "hm" }])).toBeNull();
    expect(compactionsOnly([])).toBeNull();
  });
});
