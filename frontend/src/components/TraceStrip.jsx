import React, { useState, useRef, useEffect, useMemo } from "react";
import PropTypes from "prop-types";
import { Brain, Wrench, ChevronRight, ChevronDown } from "lucide-react";
import { useTranslation } from "react-i18next";

// Live + persisted reasoning/trace strip (#90).
//
// Sits above an assistant bubble whenever the turn carried non-answer activity
// (thinking / tool_call / tool_result). One component, two lifecycles:
//
//  - LIVE turn: mounted with `live` true while the strip is the only content
//    (streaming, no answer yet) -> expanded by default, rows update as events
//    arrive. On the first answer event the parent flips `live` to false and the
//    strip auto-collapses to the one-line "Reasoning - N steps" summary. The
//    strip and the answer bubble are separate elements: nothing moves between
//    them, the strip only collapses.
//  - PERSISTED replay: mounted with `live` false -> collapsed summary the user
//    can expand. A leading {"t":"truncated"} marker renders a small
//    "(earlier steps elided)" note.
//
// Formatting rule (design SS4): NEVER raw JSON. Tool args render as
// name(pretty args); a {"raw": "..."} payload renders the raw string.

const ARROW = "→"; // -> for tool results

/**
 * Format a tool_call args object into a compact, brace-free one-liner.
 *
 *  - `{"raw": "<frag>"}`      -> the raw string as-is (unparsed args fragment)
 *  - `{}`                     -> "" (renders as name())
 *  - single key               -> just the value (e.g. calculator(2 + 2))
 *  - multiple keys            -> "k=v, k2=v2"
 *
 * String values are shown unquoted; nested objects/arrays fall back to compact
 * JSON (a value, never the whole args object).
 */
export function formatToolArgs(args) {
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    return "";
  }
  const keys = Object.keys(args);
  if (keys.length === 0) {
    return "";
  }
  if (keys.length === 1 && keys[0] === "raw") {
    return String(args.raw);
  }
  const fmt = (v) => {
    if (typeof v === "string") {
      return v;
    }
    if (v !== null && typeof v === "object") {
      return JSON.stringify(v);
    }
    return String(v);
  };
  if (keys.length === 1) {
    return fmt(args[keys[0]]);
  }
  return keys.map((k) => `${k}=${fmt(args[k])}`).join(", ");
}

/**
 * Reduce the ordered event list into renderable rows.
 *
 * Consecutive `thinking` events are merged into one streaming text block; each
 * `tool_call` and `tool_result` becomes its own row; a `truncated` marker is
 * lifted out into a flag; unknown event types are dropped (forward compat).
 * `steps` = thinking blocks + tool calls (the summary count).
 */
export function buildRows(events) {
  const list = Array.isArray(events) ? events : [];
  let truncated = false;
  const rows = [];
  for (const ev of list) {
    if (!ev || typeof ev !== "object") {
      continue;
    }
    if (ev.t === "truncated") {
      truncated = true;
    } else if (ev.t === "thinking") {
      const last = rows[rows.length - 1];
      if (last && last.kind === "thinking") {
        last.text += ev.text || "";
      } else {
        rows.push({ kind: "thinking", text: ev.text || "" });
      }
    } else if (ev.t === "tool_call") {
      rows.push({ kind: "tool_call", name: ev.name || "", argsText: formatToolArgs(ev.args) });
    } else if (ev.t === "tool_result") {
      rows.push({ kind: "tool_result", text: ev.text || "" });
    }
    // Unknown event types are intentionally ignored.
  }
  const steps = rows.filter((r) => r.kind === "thinking" || r.kind === "tool_call").length;
  return { truncated, rows, steps };
}

export default function TraceStrip({ events, live }) {
  const { t } = useTranslation();
  const { truncated, rows, steps } = useMemo(() => buildRows(events), [events]);
  const [expanded, setExpanded] = useState(live);
  const prevLive = useRef(live);
  const thinkingRef = useRef(null);

  // Auto-collapse on the live -> not-live edge only (first answer / turn end).
  // Firing only on the transition leaves a user who re-expands afterwards alone.
  useEffect(() => {
    if (prevLive.current && !live) {
      setExpanded(false);
    }
    prevLive.current = live;
  }, [live]);

  // The last thinking row is the one currently streaming; keep it pinned to the
  // bottom as text grows (internal scroll, ~6 lines).
  let lastThinkingIdx = -1;
  for (let i = rows.length - 1; i >= 0; i -= 1) {
    if (rows[i].kind === "thinking") {
      lastThinkingIdx = i;
      break;
    }
  }
  const streamingThinkingText = lastThinkingIdx >= 0 ? rows[lastThinkingIdx].text : "";
  useEffect(() => {
    if (expanded && thinkingRef.current) {
      thinkingRef.current.scrollTop = thinkingRef.current.scrollHeight;
    }
  }, [streamingThinkingText, expanded]);

  if (rows.length === 0) {
    return null;
  }

  const summary =
    steps > 0 ? t("chat:trace.summaryWithSteps", { count: steps }) : t("chat:trace.summary");

  return (
    <div className="mb-2 mr-auto w-fit max-w-[75%] rounded-xl border border-[var(--line-strong)] bg-white/[0.03] text-[var(--ink-dim)]">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        title={expanded ? t("chat:trace.hide") : t("chat:trace.show")}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs font-medium transition-colors hover:text-[var(--ink)]"
      >
        <Brain className="w-3.5 h-3.5 shrink-0" />
        <span>{summary}</span>
        {expanded ? (
          <ChevronDown className="w-3.5 h-3.5 shrink-0" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 shrink-0" />
        )}
      </button>

      {expanded && (
        <div className="flex flex-col gap-1.5 px-3 pb-2 text-xs">
          {truncated && (
            <div className="italic text-[var(--ink-faint)]">{t("chat:trace.truncated")}</div>
          )}
          {rows.map((row, i) => {
            if (row.kind === "thinking") {
              return (
                <div key={i} className="flex items-start gap-1.5">
                  <Brain className="mt-0.5 w-3.5 h-3.5 shrink-0 text-[var(--ink-faint)]" />
                  <div
                    ref={i === lastThinkingIdx ? thinkingRef : null}
                    className="max-h-24 overflow-y-auto whitespace-pre-wrap leading-relaxed"
                  >
                    {row.text}
                  </div>
                </div>
              );
            }
            if (row.kind === "tool_call") {
              return (
                <div key={i} className="flex items-start gap-1.5">
                  <Wrench className="mt-0.5 w-3.5 h-3.5 shrink-0 text-[var(--ink-faint)]" />
                  <div className="break-all font-mono">
                    {row.name}({row.argsText})
                  </div>
                </div>
              );
            }
            // tool_result: "-> <text>", clamped ~3 lines.
            return (
              <div key={i} className="line-clamp-3 break-words pl-5 text-[var(--ink-faint)]">
                {ARROW} {row.text}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

TraceStrip.propTypes = {
  // Ordered trace events: thinking / tool_call / tool_result (+ optional
  // {"t":"truncated"} marker on persisted replay). The trace's compaction
  // markers are ignored here: CompactionMarker renders them.
  events: PropTypes.arrayOf(PropTypes.object),
  // True while the strip is the only content of a live turn (expanded default);
  // flips to false on the first answer event to auto-collapse.
  live: PropTypes.bool,
};

TraceStrip.defaultProps = {
  events: [],
  live: false,
};
