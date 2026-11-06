import { describe, it, expect } from "vitest";
import { CONTEXT_GRID_BLOCKS, contextUsageView } from "./contextUsage";

// The /context panel's figures (#608): each category's share of the working
// window, and the grid of blocks that adds up to it.

const usage = (categories, window = {}, extra = {}) => ({
  categories: {
    system_prompt: 0,
    tools: 0,
    messages: 0,
    summary: 0,
    free: 0,
    autocompaction_buffer: 0,
    ...categories,
  },
  window: {
    working: 8192,
    allocated: 8192,
    trained: 40960,
    reduced_by_memory: false,
    model_loaded: true,
    ...window,
  },
  kb_excerpts_excluded: false,
  over_window: false,
  ...extra,
});

// The documented example: the six categories add up to the 8192-token window.
const DOC_EXAMPLE = {
  system_prompt: 412,
  tools: 0,
  messages: 2310,
  summary: 180,
  free: 3651,
  autocompaction_buffer: 1639,
};

const count = (blocks, key) => blocks.filter((b) => b === key).length;

describe("contextUsageView", () => {
  it("gives each category its tokens and its share of the working window", () => {
    const view = contextUsageView(usage(DOC_EXAMPLE));
    expect(view.window).toBe(8192);
    expect(view.used).toBe(412 + 2310 + 180);
    expect(view.rows.map((r) => r.key)).toEqual([
      "system_prompt",
      "tools",
      "messages",
      "summary",
      "free",
      "autocompaction_buffer",
    ]);
    const messages = view.rows.find((r) => r.key === "messages");
    expect(messages.tokens).toBe(2310);
    expect(messages.share).toBeCloseTo(2310 / 8192);
  });

  it("lists the summary only when the conversation was compacted", () => {
    const view = contextUsageView(usage({ ...DOC_EXAMPLE, summary: 0, free: 3831 }));
    expect(view.rows.map((r) => r.key)).not.toContain("summary");
  });

  it("fills the grid in legend order, the blocks adding up to the window", () => {
    const { blocks } = contextUsageView(usage(DOC_EXAMPLE));
    expect(blocks).toHaveLength(CONTEXT_GRID_BLOCKS);
    expect(count(blocks, "system_prompt")).toBe(5);
    expect(count(blocks, "tools")).toBe(0);
    expect(count(blocks, "messages")).toBe(28);
    expect(count(blocks, "summary")).toBe(2);
    expect(count(blocks, "free")).toBe(45);
    expect(count(blocks, "autocompaction_buffer")).toBe(20);
    // Used first, then free space, then the reserve.
    expect(blocks.indexOf("free")).toBe(35);
    expect(blocks.lastIndexOf("free")).toBe(79);
    expect(blocks[CONTEXT_GRID_BLOCKS - 1]).toBe("autocompaction_buffer");
  });

  it("never lets a category that holds tokens vanish from the grid", () => {
    const { blocks } = contextUsageView(
      usage({
        system_prompt: 10,
        tools: 30,
        messages: 500,
        free: 5915,
        autocompaction_buffer: 1737,
      })
    );
    expect(blocks).toHaveLength(CONTEXT_GRID_BLOCKS);
    expect(count(blocks, "system_prompt")).toBe(1);
    expect(count(blocks, "tools")).toBe(1);
  });

  it("fills every block with what is used once the request no longer fits the window", () => {
    const view = contextUsageView(
      usage({ system_prompt: 500, messages: 4000 }, { working: 4096 }, { over_window: true })
    );
    expect(view.blocks).toHaveLength(CONTEXT_GRID_BLOCKS);
    expect(count(view.blocks, "free")).toBe(0);
    expect(count(view.blocks, "autocompaction_buffer")).toBe(0);
    expect(count(view.blocks, "system_prompt") + count(view.blocks, "messages")).toBe(100);
    // Shares stay shares of the window: past it, they add up to more than 1.
    expect(view.rows.find((r) => r.key === "messages").share).toBeCloseTo(4000 / 4096);
  });

  it("says when the conversation is past its compaction threshold", () => {
    const past = contextUsageView(
      usage({ system_prompt: 400, messages: 6800, free: 0, autocompaction_buffer: 992 })
    );
    expect(past.pastThreshold).toBe(true);
    expect(contextUsageView(usage(DOC_EXAMPLE)).pastThreshold).toBe(false);
  });

  it("always draws exactly 100 blocks, even for payloads that do not add up to the window", () => {
    // A small deterministic generator: the same cases on every run.
    let seed = 42;
    const random = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const amount = () => (random() < 0.3 ? 0 : Math.floor(random() * 10 ** (1 + random() * 5)));
    for (let i = 0; i < 2000; i++) {
      const categories = {
        system_prompt: amount(),
        tools: amount(),
        messages: amount(),
        summary: amount(),
        free: amount(),
        autocompaction_buffer: amount(),
      };
      const working = 1 + Math.floor(random() * 200000);
      const { blocks, rows } = contextUsageView(usage(categories, { working }));
      const held = rows.filter((r) => r.tokens > 0);
      if (held.length === 0) {
        expect(blocks).toEqual([]);
        continue;
      }
      expect(blocks).toHaveLength(CONTEXT_GRID_BLOCKS);
      for (const r of held) {
        expect(count(blocks, r.key), JSON.stringify({ categories, working })).toBeGreaterThan(0);
      }
    }
  });

  it("fills the grid when the categories add up to far less than the window", () => {
    const { blocks } = contextUsageView(
      usage({ system_prompt: 5, messages: 15 }, { working: 1000 })
    );
    expect(blocks).toHaveLength(CONTEXT_GRID_BLOCKS);
    expect(count(blocks, "system_prompt")).toBe(25);
    expect(count(blocks, "messages")).toBe(75);
  });

  it("has no grid and no shares when no window is known", () => {
    const view = contextUsageView(
      usage({ system_prompt: 400, messages: 600 }, { working: null, trained: null })
    );
    expect(view.window).toBeNull();
    expect(view.blocks).toEqual([]);
    expect(view.rows.every((r) => r.share === null)).toBe(true);
    expect(view.pastThreshold).toBe(false);
  });
});
