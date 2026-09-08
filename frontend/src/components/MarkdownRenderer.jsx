import React from "react";
import PropTypes from "prop-types";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

// Renders markdown safely with GitHub-flavored features (tables, lists, code blocks)
// Tailwind Typography is used for nice defaults on dark backgrounds

// Local models emit inline TeX between single dollars ("$ \frac{14}{2} = 7 $",
// Qwen3 does this routinely), so single-dollar math stays ENABLED. That default
// is a currency footgun with two shapes: US-style puts the amount after the
// sign ("I have $5 and $10"), French/European style puts it before, with a
// space ("212,75 $", "1 500 $"). This guard runs AFTER remark-math and reverts
// an inline span back to literal text, using the node's source position to
// restore the exact original characters, when EITHER shape is detected: the
// opening "$" is immediately followed by a digit (US), or the source just
// before the opening "$", skipping at most one space, ends in a digit
// (French/European). Deliberate trade-off: math that starts with a bare digit
// right after the dollar ("$3x+1$") stays literal, and so does math preceded
// by "<digit><space>$" ("2 $x$") — models pad their math ("$ 3x+1 $") or write
// it without the leading dollar ("2x"), both of which render (#303, #501).
function remarkCurrencyGuard() {
  return (tree, file) => {
    const source = String(file);
    const revert = (node) => {
      if (Array.isArray(node.children)) {
        node.children.forEach(revert);
        node.children = node.children.map((child) => {
          if (child.type !== "inlineMath" || !child.position) {
            return child;
          }
          const original = source.slice(child.position.start.offset, child.position.end.offset);
          const start = child.position.start.offset;
          const preceding = source.slice(Math.max(0, start - 2), start);
          if (/^\$\d/.test(original) || /\d ?$/.test(preceding)) {
            return { type: "text", value: original, position: child.position };
          }
          return child;
        });
      }
    };
    revert(tree);
  };
}

MarkdownRenderer.propTypes = {
  content: PropTypes.string.isRequired,
  className: PropTypes.string,
};

MarkdownRenderer.defaultProps = {
  className: "",
};

export default function MarkdownRenderer({ content }) {
  return (
    <div className="prose prose-invert max-w-none whitespace-normal">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkCurrencyGuard]}
        rehypePlugins={[rehypeKatex]}
        // Do not allow raw HTML for safety in LLM outputs
        skipHtml
        components={{
          code({ node: _node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            if (inline) {
              return (
                <code className="px-1 py-0.5 rounded bg-neutral-800 text-emerald-200" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <pre className="p-3 rounded bg-neutral-900 overflow-x-auto">
                <code className={match ? `language-${match[1]}` : undefined} {...props}>
                  {children}
                </code>
              </pre>
            );
          },
          a({ children, ...props }) {
            return (
              <a
                className="text-emerald-300 underline hover:text-emerald-200"
                target="_blank"
                rel="noreferrer"
                {...props}
              >
                {children}
              </a>
            );
          },
          table({ children }) {
            return (
              <div className="overflow-x-auto">
                <table className="table-auto w-full border-collapse border border-neutral-700">
                  {children}
                </table>
              </div>
            );
          },
          th({ children }) {
            return (
              <th className="border border-neutral-700 bg-neutral-800 px-2 py-1 text-left">
                {children}
              </th>
            );
          },
          td({ children }) {
            return <td className="border border-neutral-700 px-2 py-1">{children}</td>;
          },
          ul({ children }) {
            return <ul className="list-disc pl-6">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="list-decimal pl-6">{children}</ol>;
          },
        }}
      >
        {content || ""}
      </ReactMarkdown>
    </div>
  );
}
