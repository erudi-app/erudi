// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, within } from "@testing-library/react";

import ContextUsagePanel from "./ContextUsagePanel.jsx";

// The /context panel (#608): a grid of blocks coloured by category, a legend
// with each category's estimated tokens and share of the working window, and
// the notes that explain the window it is measured against.

const usage = ({ categories = {}, window = {}, ...extra } = {}) => ({
  categories: {
    system_prompt: 412,
    tools: 0,
    messages: 2310,
    summary: 180,
    free: 3651,
    autocompaction_buffer: 1639,
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

const legend = () => screen.getByRole("list", { name: "Context usage by category" });
const row = (label) =>
  within(legend())
    .getAllByRole("listitem")
    .find((li) => li.textContent.startsWith(label));
const notes = () => screen.queryAllByRole("note").map((n) => n.textContent);

afterEach(cleanup);

describe("ContextUsagePanel", () => {
  it("draws the window as a grid of blocks, one category each", () => {
    render(<ContextUsagePanel usage={usage()} onDismiss={vi.fn()} />);
    const grid = screen.getByRole("img", { name: /Context usage/ });
    const blocks = grid.querySelectorAll("[data-category]");
    expect(blocks).toHaveLength(100);
    const of = (key) => grid.querySelectorAll(`[data-category="${key}"]`).length;
    expect(of("system_prompt")).toBe(5);
    expect(of("messages")).toBe(28);
    expect(of("summary")).toBe(2);
    expect(of("free")).toBe(45);
    expect(of("autocompaction_buffer")).toBe(20);
  });

  it("marks free space as empty blocks and the reserve distinctly", () => {
    render(<ContextUsagePanel usage={usage()} onDismiss={vi.fn()} />);
    const grid = screen.getByRole("img", { name: /Context usage/ });
    const free = grid.querySelector('[data-category="free"]').className;
    const reserve = grid.querySelector('[data-category="autocompaction_buffer"]').className;
    const used = grid.querySelector('[data-category="messages"]').className;
    expect(free).not.toMatch(/\bbg-/);
    expect(reserve).toContain("border-dashed");
    expect(used).toMatch(/\bbg-/);
    expect(new Set([free, reserve, used]).size).toBe(3);
  });

  it("lists every category with its tokens and its share, labelled as estimates", () => {
    render(<ContextUsagePanel usage={usage()} onDismiss={vi.fn()} />);
    expect(screen.getByText(/Estimates/)).toBeTruthy();
    expect(screen.getByText(/8,192/)).toBeTruthy();
    expect(row("System prompt").textContent).toContain("412 tokens");
    expect(row("System prompt").textContent).toContain("5%");
    expect(row("Tools").textContent).toContain("0 tokens");
    expect(row("Messages").textContent).toContain("2,310 tokens");
    expect(row("Messages").textContent).toContain("28.2%");
    expect(row("Summary").textContent).toContain("180 tokens");
    expect(row("Free space").textContent).toContain("3,651 tokens");
    expect(row("Auto-compaction reserve").textContent).toContain("1,639 tokens");
    expect(row("Auto-compaction reserve").textContent).toContain("20%");
  });

  it("leaves the summary out of the legend until the conversation was compacted", () => {
    render(
      <ContextUsagePanel
        usage={usage({ categories: { summary: 0, free: 3831 } })}
        onDismiss={vi.fn()}
      />
    );
    expect(row("Summary")).toBeUndefined();
  });

  it("adds no note for a loaded model whose window the memory does not bound", () => {
    render(<ContextUsagePanel usage={usage()} onDismiss={vi.fn()} />);
    expect(notes()).toEqual([]);
  });

  it("says the computer's memory bounds the window the model supports", () => {
    render(
      <ContextUsagePanel
        usage={usage({ window: { working: 6144, reduced_by_memory: true } })}
        onDismiss={vi.fn()}
      />
    );
    expect(notes()).toEqual([
      "The model supports up to 40,960 tokens; this computer's memory limits this conversation to 6,144.",
    ]);
  });

  it("says a model that is not loaded is measured against its trained window", () => {
    render(
      <ContextUsagePanel
        usage={usage({ window: { working: 40960, allocated: null, model_loaded: false } })}
        onDismiss={vi.fn()}
      />
    );
    expect(notes()).toEqual([
      "The model is not loaded: this is the window it was trained for. The window this computer allows is known once the model is loaded.",
    ]);
    expect(screen.getByText(/40,960/)).toBeTruthy();
  });

  it("says the knowledge-base excerpts a question retrieves are not counted", () => {
    render(<ContextUsagePanel usage={usage({ kb_excerpts_excluded: true })} onDismiss={vi.fn()} />);
    expect(notes()).toEqual([
      "The counts leave out the knowledge-base excerpts each question retrieves.",
    ]);
  });

  it("says the next turn will be summarized first, past the threshold", () => {
    render(
      <ContextUsagePanel
        usage={usage({
          categories: { messages: 7300, summary: 0, free: 0, autocompaction_buffer: 480 },
        })}
        onDismiss={vi.fn()}
      />
    );
    expect(row("Free space").textContent).toContain("0 tokens");
    expect(notes()).toEqual([
      "The conversation is past its compaction threshold: the next turn will summarize earlier messages first.",
    ]);
  });

  it("says so when the next request would not even fit the window", () => {
    render(
      <ContextUsagePanel
        usage={usage({
          categories: { messages: 9000, summary: 0, free: 0, autocompaction_buffer: 0 },
          over_window: true,
        })}
        onDismiss={vi.fn()}
      />
    );
    expect(notes()).toEqual([
      "The next request would not fit the model's window: the next turn will summarize earlier messages first.",
    ]);
  });

  it("shows the counts without a grid when no window is known", () => {
    render(
      <ContextUsagePanel
        usage={usage({
          categories: { free: 0, autocompaction_buffer: 0 },
          window: { working: null, allocated: null, trained: null, model_loaded: false },
        })}
        onDismiss={vi.fn()}
      />
    );
    expect(screen.queryByRole("img")).toBeNull();
    expect(row("Messages").textContent).toContain("2,310 tokens");
    expect(row("Messages").textContent).not.toContain("%");
    expect(notes()).toContain("The model's window is not known, so no share is shown.");
  });

  it("closes from its own button", () => {
    const onDismiss = vi.fn();
    render(<ContextUsagePanel usage={usage()} onDismiss={onDismiss} />);
    fireEvent.click(screen.getByRole("button", { name: "Close context usage" }));
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });
});
