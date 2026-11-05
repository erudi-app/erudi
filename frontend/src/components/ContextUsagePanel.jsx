import React from "react";
import PropTypes from "prop-types";
import { X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { contextUsageView } from "../utils/contextUsage";

// The /context panel (#608): a snapshot of the conversation's context usage,
// drawn in the thread and never stored. A grid of small blocks, one category
// each, adds up to the window the conversation works with -- what the next
// request would carry, the free space below the compaction threshold (empty
// blocks), and the auto-compaction reserve above it (dashed) -- with a legend
// giving each category's estimated tokens and share, and notes on the window.

// Presentation-only tokens kept out of the JSX tree: a block's and a legend
// swatch's look per category.
const CATEGORY_CLASS = {
  system_prompt: "bg-slate-300",
  tools: "bg-sky-400",
  messages: "bg-emerald-400",
  summary: "bg-violet-400",
  free: "border border-white/25",
  autocompaction_buffer: "border border-dashed border-amber-300/70 bg-amber-300/10",
};
// Square blocks of a fixed size (the app's Tailwind has no core aspect-ratio
// utility: the aspect-ratio plugin replaces it).
const BLOCK_SHAPE = "h-3.5 w-3.5 rounded-[3px]";
const GRID_STYLE = { gridTemplateColumns: "repeat(20, 0.875rem)" };

/** The notes that explain the window the counts are a share of, in order. */
function windowNotes(usage, view, t, number) {
  const window = usage.window ?? {};
  const notes = [];
  if (view.window === null) {
    notes.push(t("chat:context.notes.unknownWindow"));
  } else if (!window.model_loaded) {
    notes.push(t("chat:context.notes.notLoaded"));
  } else if (window.reduced_by_memory) {
    const trained = Number(window.trained);
    notes.push(
      Number.isFinite(trained) && trained > view.window
        ? t("chat:context.notes.reducedByMemory", {
            trained: number(trained),
            working: number(view.window),
          })
        : t("chat:context.notes.reducedByMemoryNoTrained", { working: number(view.window) })
    );
  }
  if (usage.kb_excerpts_excluded) {
    notes.push(t("chat:context.notes.kbExcluded"));
  }
  if (usage.over_window) {
    notes.push(t("chat:context.notes.overWindow"));
  } else if (view.pastThreshold) {
    notes.push(t("chat:context.notes.pastThreshold"));
  }
  return notes;
}

export default function ContextUsagePanel({ usage, onDismiss }) {
  const { t, i18n } = useTranslation();
  const view = contextUsageView(usage);
  const numberFormat = new Intl.NumberFormat(i18n.language);
  const shareFormat = new Intl.NumberFormat(i18n.language, {
    style: "percent",
    maximumFractionDigits: 1,
  });
  const number = (n) => numberFormat.format(n);
  const notes = windowNotes(usage, view, t, number);

  return (
    <section
      aria-label={t("chat:context.title")}
      className="relative mx-auto w-full max-w-xl rounded-2xl border border-white/10 bg-[rgba(22,40,36,0.45)] p-4 text-sm text-gray-200"
    >
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h4 className="font-semibold text-[#F2F7F4]">{t("chat:context.title")}</h4>
          <p className="text-xs text-gray-400">
            {view.window === null
              ? t("chat:context.estimatesNoWindow")
              : t("chat:context.estimates", { window: number(view.window) })}
          </p>
        </div>
        <button
          type="button"
          onClick={onDismiss}
          aria-label={t("chat:context.dismiss")}
          className="inline-flex shrink-0 items-center justify-center w-7 h-7 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-gray-300 hover:text-gray-100 transition"
        >
          <X className="w-3.5 h-3.5" aria-hidden />
        </button>
      </div>

      {view.blocks.length > 0 && (
        <div
          role="img"
          aria-label={t("chat:context.gridLabel", {
            used: number(view.used),
            window: number(view.window),
          })}
          className="mb-3 grid gap-[3px]"
          style={GRID_STYLE}
        >
          {view.blocks.map((category, i) => (
            <span
              key={i}
              data-category={category}
              className={`${BLOCK_SHAPE} ${CATEGORY_CLASS[category]}`}
            />
          ))}
        </div>
      )}

      <ul aria-label={t("chat:context.legendLabel")} className="flex flex-col gap-1 text-xs">
        {view.rows.map((row) => (
          <li key={row.key} className="flex items-center gap-2">
            <span aria-hidden className={`shrink-0 ${BLOCK_SHAPE} ${CATEGORY_CLASS[row.key]}`} />
            <span className="flex-1 text-gray-200">{t(`chat:context.categories.${row.key}`)}</span>
            <span className="tabular-nums text-gray-400">
              {row.share === null
                ? t("chat:context.valueNoShare", { count: row.tokens, tokens: number(row.tokens) })
                : t("chat:context.value", {
                    count: row.tokens,
                    tokens: number(row.tokens),
                    share: shareFormat.format(row.share),
                  })}
            </span>
          </li>
        ))}
      </ul>

      {notes.map((note) => (
        <p key={note} role="note" className="mt-2 text-xs text-gray-400">
          {note}
        </p>
      ))}
    </section>
  );
}

ContextUsagePanel.propTypes = {
  // The GET /conversations/{id}/context answer.
  usage: PropTypes.shape({
    categories: PropTypes.object.isRequired,
    window: PropTypes.shape({
      working: PropTypes.number,
      allocated: PropTypes.number,
      trained: PropTypes.number,
      reduced_by_memory: PropTypes.bool,
      model_loaded: PropTypes.bool,
    }).isRequired,
    kb_excerpts_excluded: PropTypes.bool,
    over_window: PropTypes.bool,
  }).isRequired,
  onDismiss: PropTypes.func.isRequired,
};
