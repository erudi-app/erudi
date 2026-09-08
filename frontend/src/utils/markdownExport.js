/**
 * Formats a full conversation object into a Markdown string, suitable for exporting.
 * 
 * @param {Object} conv - The conversation object (with messages).
 * @returns {string} - The formatted Markdown string.
 */
export function formatConversationToMarkdown(conv) {
  if (!conv) return "";

  const dateStr = new Date(conv.created_at).toLocaleString();
  const name = conv.name || "Conversation";
  const llm = conv.llm_id || "Unknown Model";

  let md = `# ${name}\n\n`;
  md += `- **Model**: ${llm}\n`;
  md += `- **Date**: ${dateStr}\n\n`;
  md += `---\n\n`;

  if (Array.isArray(conv.messages)) {
    conv.messages.forEach(msg => {
      // Use role "User" or "Assistant" for headings
      const roleStr = msg.role === 'user' ? 'User' : 'Assistant';
      md += `## ${roleStr}\n\n`;
      
      let content = msg.content || "";
      
      // Filter out <think>...</think> reasoning traces, including newlines
      content = content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
      
      if (content) {
        md += `${content}\n\n`;
      }
    });
  }

  return md;
}
