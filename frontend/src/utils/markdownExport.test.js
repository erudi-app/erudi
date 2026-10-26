import { describe, it, expect } from "vitest";
import { formatConversationToMarkdown } from "./markdownExport";

describe("formatConversationToMarkdown", () => {
  it("formats a conversation with basic metadata", () => {
    const conv = {
      id: 42,
      name: "Test Conversation",
      llm_id: "test-model-v1",
      created_at: "2026-09-03T10:00:00Z",
      messages: []
    };

    const md = formatConversationToMarkdown(conv);
    expect(md).toContain("# Test Conversation");
    expect(md).toContain("- **Model**: test-model-v1");
    expect(md).toContain("- **Date**:"); // Exact format depends on locale, so just check label
    expect(md).toContain("---");
  });

  it("formats user and assistant messages properly", () => {
    const conv = {
      created_at: "2026-09-03T10:00:00Z",
      messages: [
        { role: "user", content: "Hello world" },
        { role: "assistant", content: "Hi there!" }
      ]
    };

    const md = formatConversationToMarkdown(conv);
    expect(md).toContain("## User\n\nHello world\n\n");
    expect(md).toContain("## Assistant\n\nHi there!\n\n");
  });

  it("filters out reasoning traces (<think> blocks)", () => {
    const conv = {
      created_at: "2026-09-03T10:00:00Z",
      messages: [
        { role: "user", content: "Solve 2+2" },
        { role: "assistant", content: "<think>\n2 plus 2 is 4.\n</think>\nThe answer is 4." }
      ]
    };

    const md = formatConversationToMarkdown(conv);
    expect(md).not.toContain("2 plus 2 is 4.");
    expect(md).not.toContain("<think>");
    expect(md).toContain("## Assistant\n\nThe answer is 4.\n\n");
  });

  it("handles missing or malformed data gracefully", () => {
    expect(formatConversationToMarkdown(null)).toBe("");
    expect(formatConversationToMarkdown({})).toContain("Conversation");
  });
});
