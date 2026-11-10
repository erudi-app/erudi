import React, { useMemo } from "react";
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
// sign ("I have $5 and $10"), French/European style puts it before, separated
// by one of three spaces models are seen to emit — the ASCII space, the
// no-break space U+00A0, or French typography's narrow no-break space U+202F
// ("212,75 $", "1 500 $"). This guard runs AFTER remark-math and reverts
// an inline span back to literal text, using the node's source position to
// restore the exact original characters, when EITHER shape is detected: the
// opening "$" is immediately followed by a digit (US), or the source just
// before the opening "$", skipping at most one of those three separators, ends
// in a digit (French/European). Deliberate trade-off: math that starts with a
// bare digit right after the dollar ("$3x+1$") stays literal, and so does math
// preceded by "<digit><space>$" ("2 $x$") — models pad their math
// ("$ 3x+1 $") or write it without the leading dollar ("2x"), both of which
// render (#303, #501).
//
// Math that arrives as \( ... \) is converted to "$ ... $" before parsing
// (convertTexDelimiters), and the guard must not read such a "$" as currency:
// "\(3x\)" becomes "$3x$" and "2 \(x\)" becomes "2 $x$", both of which it
// would revert. The conversion reports where each converted opening "$" sits
// in the converted text (`trusted`); the guard leaves those spans alone, and
// still judges every "$" the model wrote itself.
function remarkCurrencyGuard({ trusted = new Set() } = {}) {
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
          if (trusted.has(child.position.start.offset)) {
            return child;
          }
          if (/^\$\d/.test(original) || /\d[ \u00A0\u202F]?$/.test(preceding)) {
            return { type: "text", value: original, position: child.position };
          }
          return child;
        });
      }
    };
    revert(tree);
  };
}

// A closing fence: the same character as the opening one, at least as long.
const FENCE_OPEN = /^ {0,3}(`{3,}|~{3,})/;
// An inline code span: a run of backticks, up to the next run of the same length.
const INLINE_CODE = /(?<!`)(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)/g;
// \( ... \) stays on one paragraph; \[ ... \] may span lines. An opener
// preceded by a backslash is TeX's own "\\" and is left alone.
const TEX_DELIMITERS = /(?<!\\)\\\(((?:[^\n]|\n(?!\s*\n))+?)\\\)|(?<!\\)\\\[([\s\S]+?)\\\]/g;

// Models such as Qwen2.5 write math between \( ... \) and \[ ... \]. Markdown
// reads "\(" as an escaped parenthesis before remark-math sees it, so the
// delimiters become "$ ... $" and "$$ ... $$" in the raw text, never inside a
// fenced block or an inline code span. An unclosed delimiter (a streaming
// answer cut mid-formula) matches nothing and stays text until it closes.
// Returns the converted text and `trusted`, the offsets in it of the "$" that
// open a converted inline span (see remarkCurrencyGuard).
export function convertTexDelimiters(text) {
  const trusted = new Set();
  let out = "";

  // `lastSegment`: no inline code follows this prose in its chunk.
  const convertProse = (prose, lastSegment) => {
    let last = 0;
    for (const match of prose.matchAll(TEX_DELIMITERS)) {
      out += prose.slice(last, match.index);
      last = match.index + match[0].length;
      if (match[1] !== undefined) {
        trusted.add(out.length);
        out += `$${match[1]}$`;
      } else if (
        /(^|\n)[ \t]*$/.test(out) &&
        (/^[ \t]*\n/.test(prose.slice(last)) || (lastSegment && /^[ \t]*$/.test(prose.slice(last))))
      ) {
        // Alone on its lines: remark-math reads a block only when each "$$"
        // has its own line, and only a block renders as a display formula.
        out += `$$\n${match[2].trim()}\n$$`;
      } else {
        out += `$$${match[2]}$$`;
      }
    }
    out += prose.slice(last);
  };

  const convertChunk = (prose) => {
    let last = 0;
    for (const match of prose.matchAll(INLINE_CODE)) {
      convertProse(prose.slice(last, match.index), false);
      out += match[0];
      last = match.index + match[0].length;
    }
    convertProse(prose.slice(last), true);
  };

  // Walk the lines: a fenced block (an unclosed one runs to the end) is
  // copied as is; the prose between fences is converted as one chunk, so a
  // "\[" may span lines.
  const lines = text.split("\n");
  let prose = [];
  let fence = null;
  // `terminated`: a fence follows, so the chunk's last line ends in a newline.
  const flushProse = (terminated) => {
    if (prose.length > 0) {
      convertChunk(prose.join("\n"));
      if (terminated) {
        out += "\n";
      }
      prose = [];
    }
  };
  lines.forEach((line, i) => {
    const eol = i < lines.length - 1 ? "\n" : "";
    if (fence) {
      out += line + eol;
      const close = new RegExp(`^ {0,3}${fence[0]}{${fence.length},}[ \\t]*$`);
      if (close.test(line)) {
        fence = null;
      }
      return;
    }
    const open = FENCE_OPEN.exec(line);
    if (open) {
      flushProse(true);
      fence = open[1];
      out += line + eol;
      return;
    }
    prose.push(line);
  });
  flushProse(false);
  return { text: out, trusted };
}

MarkdownRenderer.propTypes = {
  content: PropTypes.string.isRequired,
  className: PropTypes.string,
};

MarkdownRenderer.defaultProps = {
  className: "",
};

export default function MarkdownRenderer({ content }) {
  const converted = useMemo(() => convertTexDelimiters(content || ""), [content]);
  const remarkPlugins = useMemo(
    () => [remarkGfm, remarkMath, [remarkCurrencyGuard, { trusted: converted.trusted }]],
    [converted]
  );
  return (
    <div className="prose prose-invert max-w-none whitespace-normal">
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
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
        {converted.text}
      </ReactMarkdown>
    </div>
  );
}
