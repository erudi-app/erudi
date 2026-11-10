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
// Math that arrives as \( ... \) or \[ ... \] is converted to "$$ ... $$"
// before parsing (convertTexDelimiters), and the guard must not read such a
// span as currency: "\(3x\)" becomes "$$3x$$" and "2 \(x\)" becomes
// "2 $$x$$", the second of which it would revert. The conversion reports where
// each converted opener sits in the converted text (`trusted`); the guard
// leaves those spans alone, and still judges every "$" the model wrote itself.
// Known limit: two lone currency "$" around a converted span can still pair
// with each other, as they can around any hand-written "$x$".
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

// A fence opens with three or more backticks or tildes; it closes on the same
// character, at least as long.
const FENCE_OPEN = /^ {0,3}(`{3,}|~{3,})/;
// What conversion never enters, inside a line chunk: an inline code span (a
// run of backticks up to the next run of the same length, never across a
// blank line) and a link destination "](...)".
const PROTECTED = /(?<!`)(`+)(?!`)(?:[^\n]|\n(?![ \t]*\n))*?(?<!`)\1(?!`)|\]\([^)\n]*\)/g;
// \( ... \) stays on one paragraph; \[ ... \] may span lines. An opener
// preceded by a backslash is TeX's own "\\" and is left alone.
const TEX_DELIMITERS = /(?<!\\)\\\(((?:[^\n]|\n(?!\s*\n))+?)\\\)|(?<!\\)\\\[([\s\S]+?)\\\]/g;
const LIST_ITEM = /^[ \t]*(?:[-*+]|\d+[.)])[ \t]/;
const INDENTED = /^(?: {4}|\t)/;

// Models such as Qwen2.5 write math between \( ... \) and \[ ... \]. Markdown
// reads "\(" as an escaped parenthesis before remark-math sees it, so the
// delimiters become "$$ ... $$" in the raw text. Never converted: fenced
// blocks (an unclosed one runs to the end), 4-space indented code blocks
// outside a list, inline code, link destinations, a "\[ ... \]" followed by
// "(" (not math) and one holding only digits and commas (a citation such as
// "\[1\]"). An unclosed delimiter (a streaming answer cut mid-formula) matches
// nothing and stays text until it closes. Known limit: a path such as
// "C:\(x86)" with a later ")" is read as math.
//
// Inline spans take double dollars: remark-math pairs a "$$" only with a "$$",
// so a lone currency "$" in the same paragraph never pairs with them. A
// "\[ ... \]" alone on its lines becomes a block ("$$" on lines of its own,
// repeating the line's list or quote prefix so the block stays inside its
// item), the only form that renders as a display formula.
// Returns the converted text and `trusted`, the offsets in it of every "$$"
// that opens an inline span (see remarkCurrencyGuard).
export function convertTexDelimiters(text) {
  const trusted = new Set();
  let out = "";

  // `lastSegment`: no protected span follows this prose in its chunk.
  const convertProse = (prose, lastSegment) => {
    let last = 0;
    for (const match of prose.matchAll(TEX_DELIMITERS)) {
      const rest = prose.slice(match.index + match[0].length);
      const display = match[1] === undefined;
      const body = display ? match[2] : match[1];
      if (display && (/^\(/.test(rest) || /^[\d,\s]*$/.test(body))) {
        continue;
      }
      out += prose.slice(last, match.index);
      last = match.index + match[0].length;
      const prefix = /(?:^|\n)([ \t]*(?:>[ \t]*)*)$/.exec(out)?.[1];
      const alone = /^[ \t]*\n/.test(rest) || (lastSegment && /^[ \t]*$/.test(rest));
      if (display && prefix !== undefined && alone) {
        out += `$$\n${prefix}${body.trim()}\n${prefix}$$`;
      } else {
        trusted.add(out.length);
        out += `$$${body}$$`;
      }
    }
    out += prose.slice(last);
  };

  const convertChunk = (prose) => {
    let last = 0;
    for (const match of prose.matchAll(PROTECTED)) {
      convertProse(prose.slice(last, match.index), false);
      out += match[0];
      last = match.index + match[0].length;
    }
    convertProse(prose.slice(last), true);
  };

  // Walk the lines: code is copied as is; the prose between code is
  // converted as one chunk, so a "\[" may span lines.
  const lines = text.split("\n");
  let prose = [];
  let fence = null;
  let inList = false;
  let prevBlank = true;
  let prevCode = false;
  // `terminated`: code follows, so the chunk's last line ends in a newline.
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
      prevBlank = false;
      prevCode = true;
      return;
    }
    const blank = line.trim() === "";
    if (!blank && !inList && INDENTED.test(line) && (prevBlank || prevCode)) {
      flushProse(true);
      out += line + eol;
      prevBlank = false;
      prevCode = true;
      return;
    }
    if (LIST_ITEM.test(line)) {
      inList = true;
    } else if (!blank && !/^[ \t]/.test(line)) {
      inList = false;
    }
    prevBlank = blank;
    prevCode = false;
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
