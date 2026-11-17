import { describe, it, expect } from "vitest";
import { buildConversationMarkdown } from "./conversationMarkdown";

// Export of one conversation as Markdown (#430): a pure function of the
// conversation the backend returns and the labels the caller translated.

const labels = {
  untitled: "Untitled conversation",
  user: "You",
  assistant: "Assistant",
  attachment: "Attachment",
  image: "Image",
  created: "Created Nov 3, 2026",
  compaction: (count) =>
    count > 0
      ? `${count} earlier messages were summarized here.`
      : "Earlier messages were summarized here.",
};

const msg = (sender, content, trace = null) => ({ id: Math.random(), sender, content, trace });

const compactionDone = (extra = {}) => ({
  t: "compaction",
  phase: "done",
  summarized: 4,
  reason: "length",
  summary: "THE SUMMARY TEXT",
  ...extra,
});

describe("buildConversationMarkdown", () => {
  it("writes the title, the date and each message under a role heading, in order", () => {
    const md = buildConversationMarkdown(
      { name: "Airships", messages: [msg("user", "Hello"), msg("llm", "Hi there!")] },
      labels
    );
    expect(md).toBe(
      "# Airships\n\nCreated Nov 3, 2026\n\n## You\n\nHello\n\n## Assistant\n\nHi there!\n"
    );
  });

  it("shows the translated untitled name for a conversation that has no title", () => {
    const md = buildConversationMarkdown({ name: "New Conversation", messages: [] }, labels);
    expect(md.startsWith("# Untitled conversation\n")).toBe(true);
  });

  it("keeps a title on one heading line", () => {
    const md = buildConversationMarkdown({ name: "Two\nlines", messages: [] }, labels);
    expect(md.startsWith("# Two lines\n")).toBe(true);
  });

  it("leaves the model's reasoning out", () => {
    const md = buildConversationMarkdown(
      {
        name: "T",
        messages: [
          msg("user", "2+2?"),
          msg("llm", "4", [
            { t: "thinking", text: "SECRET REASONING" },
            { t: "tool_call", name: "search" },
          ]),
        ],
      },
      labels
    );
    expect(md).toContain("4");
    expect(md).not.toContain("SECRET REASONING");
    expect(md).not.toContain("search");
  });

  it("leaves a failed turn out, generic or curated", () => {
    const md = buildConversationMarkdown(
      {
        name: "T",
        messages: [
          msg("user", "hello"),
          msg("llm", "[ERROR_MESSAGE_SYSTEM] I apologize, but I encountered an error"),
          msg("user", "again"),
          msg("llm", "[ERROR_MESSAGE_SYSTEM] The model ran out of memory", [
            { t: "curated_error", kind: "oom" },
          ]),
        ],
      },
      labels
    );
    expect(md).not.toContain("ERROR_MESSAGE_SYSTEM");
    expect(md).not.toContain("apologize");
    expect(md).not.toContain("ran out of memory");
    expect(md.match(/## Assistant/g)).toBeNull();
    expect(md.match(/## You/g)).toHaveLength(2);
  });

  it("marks an automatic compaction with a separator before the answer it preceded", () => {
    const md = buildConversationMarkdown(
      { name: "T", messages: [msg("user", "q"), msg("llm", "answer", [compactionDone()])] },
      labels
    );
    expect(md).toContain(
      "## You\n\nq\n\n---\n\n*4 earlier messages were summarized here.*\n\n---\n\n## Assistant\n\nanswer\n"
    );
  });

  it("marks a /compact run after the answer it was saved on", () => {
    const md = buildConversationMarkdown(
      {
        name: "T",
        messages: [msg("llm", "answer", [compactionDone({ manual: true, summarized: 0 })])],
      },
      labels
    );
    expect(md).toContain(
      "## Assistant\n\nanswer\n\n---\n\n*Earlier messages were summarized here.*\n\n---\n"
    );
  });

  it("never writes the summary the model keeps as its working memory", () => {
    const md = buildConversationMarkdown(
      { name: "T", messages: [msg("llm", "answer", [compactionDone()])] },
      labels
    );
    expect(md).not.toContain("THE SUMMARY TEXT");
  });

  it("still marks a compaction that a failed turn carries, without the failed turn", () => {
    const md = buildConversationMarkdown(
      {
        name: "T",
        messages: [
          msg("user", "q"),
          msg("llm", "[ERROR_MESSAGE_SYSTEM] boom", [
            compactionDone(),
            { t: "error_detail", text: "cause" },
          ]),
        ],
      },
      labels
    );
    expect(md).toContain("*4 earlier messages were summarized here.*");
    expect(md).not.toContain("boom");
    expect(md).not.toContain("cause");
  });

  it("names attachments and images instead of embedding them", () => {
    const content =
      "Look at these\n[image_path:/home/me/pics/cat%5D.png][image][file_path:/docs/report.pdf]";
    const md = buildConversationMarkdown({ name: "T", messages: [msg("user", content)] }, labels);
    expect(md).toContain("## You\n\nLook at these\n\n");
    expect(md).toContain("> Image: cat].png\n");
    expect(md).toContain("> Image\n");
    expect(md).toContain("> Attachment: report.pdf\n");
    expect(md).not.toContain("/home/me");
    expect(md).not.toContain("[image");
    expect(md).not.toContain("[file_path");
  });

  it("keeps a message that is only attachments", () => {
    const md = buildConversationMarkdown(
      { name: "T", messages: [msg("user", "[file_path:/a/notes.txt]")] },
      labels
    );
    expect(md).toContain("## You\n\n> Attachment: notes.txt\n");
  });

  describe("a message body cannot break the structure around it", () => {
    it("closes a code fence the message left open, before the next turn", () => {
      const md = buildConversationMarkdown(
        {
          name: "T",
          messages: [msg("llm", "Here:\n```js\nconst a = 1;"), msg("user", "thanks")],
        },
        labels
      );
      expect(md).toContain("```js\nconst a = 1;\n```\n\n## You\n\nthanks\n");
    });

    it("closes a fence with the marker that opened it", () => {
      const tilde = buildConversationMarkdown(
        { name: "T", messages: [msg("llm", "~~~\ncode"), msg("user", "next")] },
        labels
      );
      expect(tilde).toContain("~~~\ncode\n~~~\n\n## You");
      const long = buildConversationMarkdown(
        { name: "T", messages: [msg("llm", "````md\n```\ninner"), msg("user", "next")] },
        labels
      );
      expect(long).toContain("````md\n```\ninner\n````\n\n## You");
    });

    it("leaves a balanced fence alone", () => {
      const body = "```py\nprint(1)\n```\nafter";
      const md = buildConversationMarkdown({ name: "T", messages: [msg("llm", body)] }, labels);
      expect(md).toContain(`## Assistant\n\n${body}\n`);
    });

    it("closes a fence an attachment note would otherwise end up inside", () => {
      const md = buildConversationMarkdown(
        { name: "T", messages: [msg("user", "```\nopen[file_path:/a/b.txt]")] },
        labels
      );
      expect(md).toContain("```\nopen\n```\n\n> Attachment: b.txt\n");
    });

    it("moves the headings of a body below the role heading", () => {
      const md = buildConversationMarkdown(
        {
          name: "T",
          messages: [msg("llm", "# Title\n\ntext\n\n## You\n\n###### deep")],
        },
        labels
      );
      expect(md).toContain("### Title\n");
      expect(md).toContain("### You\n");
      expect(md).toContain("###### deep\n");
      expect(md.match(/^## /gm)).toEqual(["## "]);
      expect(md.match(/^# /gm)).toEqual(["# "]);
    });

    it("does not touch a hash line inside a code fence or one that is no heading", () => {
      const body = "```sh\n# a comment\n```\n#hashtag\n    # indented code";
      const md = buildConversationMarkdown({ name: "T", messages: [msg("llm", body)] }, labels);
      expect(md).toContain(`## Assistant\n\n${body}\n`);
    });
  });

  it("writes only the title for a conversation without messages or anything unexpected", () => {
    expect(buildConversationMarkdown({ name: "T" }, { ...labels, created: "" })).toBe("# T\n");
    expect(buildConversationMarkdown(null, labels)).toBe("");
  });
});
