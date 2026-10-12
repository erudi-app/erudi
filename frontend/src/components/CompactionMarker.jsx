import React, { useState } from "react";
import PropTypes from "prop-types";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";

// Where a compaction took over (#614).
//
// One discreet separator per compaction the turn ran (the `done` events of its
// trace), stacked in order above the turn's reasoning strip and answer: a thin
// rule with how many earlier messages were summarized and why -- the
// conversation outgrew the model's context (`length`) or this computer's
// memory bounds it (`memory`). The summary is readable on demand, as plain
// text (never markdown: it is the model's working memory, shown as written).
// A lost summary says the earlier messages were dropped, and offers the
// previous summary the model still keeps, when there is one.

const REASON_KEYS = {
  length: { counted: "chat:compaction.length", uncounted: "chat:compaction.lengthNoCount" },
  memory: { counted: "chat:compaction.memory", uncounted: "chat:compaction.memoryNoCount" },
};

function reasonKeys(reason) {
  return Object.prototype.hasOwnProperty.call(REASON_KEYS, reason)
    ? REASON_KEYS[reason]
    : REASON_KEYS.length;
}

function CompactionSeparator({ event }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const count = Number.isInteger(event.summarized) && event.summarized > 0 ? event.summarized : 0;
  const keys = reasonKeys(event.reason);
  const lost = event.summary_lost === true;
  let label;
  if (lost) {
    label = t("chat:compaction.lost");
  } else if (count > 0) {
    label = t(keys.counted, { count });
  } else {
    label = t(keys.uncounted);
  }
  const summary = typeof event.summary === "string" ? event.summary.trim() : "";
  const showLabel = lost ? t("chat:compaction.showKeptSummary") : t("chat:compaction.showSummary");

  return (
    <div className="flex flex-col gap-1 text-xs text-[var(--ink-faint)]">
      <div className="flex items-center gap-3">
        <span aria-hidden="true" className="h-px flex-1 bg-[var(--line)]" />
        <span className="text-center">{label}</span>
        <span aria-hidden="true" className="h-px flex-1 bg-[var(--line)]" />
      </div>
      {summary && (
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="mx-auto flex items-center gap-1 transition-colors hover:text-[var(--ink)]"
        >
          {open ? (
            <ChevronDown className="h-3.5 w-3.5 shrink-0" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 shrink-0" />
          )}
          <span>{open ? t("chat:compaction.hideSummary") : showLabel}</span>
        </button>
      )}
      {summary && open && (
        <div className="mx-auto max-w-[75%] whitespace-pre-wrap rounded-xl border border-[var(--line-strong)] bg-white/[0.03] px-3 py-2 leading-relaxed text-[var(--ink-dim)]">
          {summary}
        </div>
      )}
    </div>
  );
}

CompactionSeparator.propTypes = {
  event: PropTypes.shape({
    summarized: PropTypes.number,
    reason: PropTypes.string,
    summary: PropTypes.string,
    summary_lost: PropTypes.bool,
  }).isRequired,
};

export default function CompactionMarker({ events }) {
  if (!Array.isArray(events) || events.length === 0) {
    return null;
  }
  // Keyed by the event's index within the message: each separator keeps its
  // own open/closed state while the turn's trace grows.
  return (
    <div className="mb-2 flex flex-col gap-2 self-stretch">
      {events.map((event, i) => (
        <CompactionSeparator key={i} event={event} />
      ))}
    </div>
  );
}

CompactionMarker.propTypes = {
  // The turn's {"t":"compaction","phase":"done",...} events, in order.
  events: PropTypes.arrayOf(PropTypes.object),
};
