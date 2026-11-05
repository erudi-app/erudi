import { describe, it, expect, vi } from "vitest";
import {
  compactingVersion,
  finishCompacting,
  isCompacting,
  startCompacting,
  subscribeCompacting,
} from "./compactingConversations";

// The conversations a /compact runs for (#585): one run per conversation,
// shared by every page instance of the session.

describe("compactingConversations", () => {
  it("runs one /compact per conversation at a time", () => {
    expect(startCompacting("7")).toBe(true);
    expect(isCompacting(7)).toBe(true);
    expect(startCompacting(7)).toBe(false);
    expect(startCompacting("8")).toBe(true);
    finishCompacting("7");
    expect(isCompacting("7")).toBe(false);
    expect(isCompacting("8")).toBe(true);
    finishCompacting("8");
  });

  it("tells its listeners when a run starts and ends, and bumps its version", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeCompacting(listener);
    const before = compactingVersion();
    startCompacting("9");
    finishCompacting("9");
    finishCompacting("9"); // not running any more: no news
    expect(listener.mock.calls).toEqual([["9"], ["9"]]);
    expect(compactingVersion()).toBe(before + 2);
    unsubscribe();
    startCompacting("9");
    finishCompacting("9");
    expect(listener).toHaveBeenCalledTimes(2);
  });
});
