import React, { useState, useMemo, useEffect } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import { Search } from "lucide-react";
import ExploreModelCard from "./ExploreModelCard";
import { rankByFit } from "../utils/hardwareFit";
import apiClient from "../services/api/client";
import { createLogger } from "../utils/logger";

const log = createLogger("HuggingFaceSearch");

// Starting points that map to real capabilities people look for. These are the
// literal query terms sent to Hugging Face (whose model names and tags are in
// English), so they are deliberately not translated.
const SUGGESTIONS = ["coding", "reasoning", "vision", "tiny", "uncensored", "multilingual"];

const SORT_KEYS = ["fit", "downloads", "likes", "smallest", "largest"];

/**
 * Live Hugging Face search, reframed as a first-class discovery tool (#122):
 * it searches all of HF — beyond the curated catalog — but only for models that
 * run on this machine's engine, and ranks hits by how well they fit the hardware
 * budget. Empty and error states give direction rather than mood.
 *
 * `handoff` is how the offline catalog search escalates (#380): when nothing in
 * the bundled catalog matches, the page hands the term over as
 * `{ term, seq }` and this panel runs it as if the user had typed it here. `seq`
 * increments on every handoff so the same term can be handed over twice.
 */
export default function HuggingFaceSearchPanel({
  range,
  onDownload,
  onInfo,
  isInstalled,
  handoff,
}) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null); // null = not searched yet
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchedTerm, setSearchedTerm] = useState("");
  const [collapsed, setCollapsed] = useState(false);
  const [sortBy, setSortBy] = useState("fit");

  const sortedResults = useMemo(() => {
    if (!results) return results;
    const r = [...results];
    if (sortBy === "fit") return r; // already ranked by fit from rankByFit
    if (sortBy === "downloads") return r.sort((a, b) => (b.downloads || 0) - (a.downloads || 0));
    if (sortBy === "likes") return r.sort((a, b) => (b.likes || 0) - (a.likes || 0));
    if (sortBy === "smallest")
      return r.sort((a, b) => (a.param_size || Infinity) - (b.param_size || Infinity));
    if (sortBy === "largest") return r.sort((a, b) => (b.param_size || 0) - (a.param_size || 0));
    return r;
  }, [results, sortBy]);

  const clear = () => {
    setResults(null);
    setError("");
    setSearchedTerm("");
  };

  // The synchronous half of a search: the panel opens on the term, the sort
  // goes back to fit, and the offline guard answers. Searching Hugging Face
  // needs the internet, so say so plainly instead of returning an empty "no
  // matches" or throwing. Returns whether the request has to go out.
  const beginSearch = (q) => {
    setCollapsed(false);
    setQuery(q);
    setSearchedTerm(q);
    setError("");
    setSortBy("fit");
    if (typeof navigator !== "undefined" && navigator.onLine === false) {
      setResults([]);
      setError(t("models:search.offline"));
      return false;
    }
    setLoading(true);
    return true;
  };

  const runSearch = async (term) => {
    const q = (term ?? query).trim();
    if (!q) {
      return;
    }
    if (!beginSearch(q)) return;
    await fetchHits(q);
  };

  const fetchHits = async (q) => {
    try {
      const data = await apiClient.get(`/llms/search/huggingface?q=${encodeURIComponent(q)}`);
      const mapped = data.map((m) => ({
        name: m.name,
        link: m.link,
        category: m.category,
        param_size: m.param_size,
        quantized: m.quantized,
        gated: m.gated,
        runnable: true,
        // Details-modal fields, from the search hit. "Unknown" is the sentinel
        // the details modal compares against to hide an absent field.
        parameters: m.param_size ? `${m.param_size}B` : "Unknown",
        downloads: m.downloads ? String(m.downloads) : "Unknown",
        likes: m.likes ? String(m.likes) : "Unknown",
        pipeline: m.pipeline_tag || "Unknown",
      }));
      setResults(rankByFit(mapped, range));
    } catch (err) {
      log.error("HF search failed:", err);
      // A failed fetch here is a request error (timeout, HF unreachable, 5xx),
      // not proof the machine is offline — the genuine-offline case is caught by
      // the navigator.onLine guard above. Don't mislabel it as "no internet" (#210).
      setError(t("models:search.unreachable"));
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  // A handoff is a new object each time: its synchronous half is applied while
  // rendering, and the request goes out from the effect below.
  const [seenHandoff, setSeenHandoff] = useState(null);
  const [handoffRequest, setHandoffRequest] = useState(null);
  if (handoff !== seenHandoff) {
    setSeenHandoff(handoff);
    const q = handoff?.term?.trim();
    if (q && beginSearch(q)) setHandoffRequest({ q });
  }

  useEffect(() => {
    if (handoffRequest) fetchHits(handoffRequest.q);
    // fetchHits is recreated every render; the request object is the trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handoffRequest]);

  return (
    <section className="rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-5">
      <span className="eyebrow !text-[var(--fit-good)]">{t("models:search.title")}</span>
      <p className="text-[13px] text-[var(--ink-dim)] mt-1.5 mb-4 max-w-xl">
        {t("models:search.description")}
      </p>

      {/* Search field */}
      <div className="flex items-center gap-2 rounded-xl border border-[var(--line-strong)] bg-[var(--canvas)] px-3 py-2.5 focus-within:border-[var(--fit-good)] transition-colors">
        <Search className="w-4 h-4 text-[var(--ink-faint)] flex-shrink-0" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && runSearch()}
          placeholder={t("models:search.placeholder")}
          className="flex-1 bg-transparent border-0 text-sm text-[var(--ink)] placeholder-[var(--ink-faint)] focus:outline-none focus:ring-0"
        />
        <button
          onClick={() => runSearch()}
          disabled={!query.trim() || loading}
          className="rounded-lg bg-[var(--fit-good)] text-[#07241d] px-4 py-1.5 text-[13px] font-medium transition-[filter] hover:brightness-110 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          {loading ? t("models:search.searching") : t("common:actions.search")}
        </button>
      </div>

      {/* Suggestion chips */}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="mono text-[11px] text-[var(--ink-faint)] mr-1">
          {t("models:search.tryLabel")}
        </span>
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => runSearch(s)}
            className="mono text-[11px] rounded-full border border-[var(--line)] px-2.5 py-1 text-[var(--ink-dim)] hover:text-[var(--ink)] hover:border-[var(--line-strong)] transition-colors"
          >
            {s}
          </button>
        ))}
      </div>

      {/* Results */}
      {(loading || results !== null) && (
        <div className="mt-5 pt-5 border-t border-[var(--line)]">
          {loading ? (
            <div className="flex items-center gap-2 text-[var(--ink-faint)] mono text-xs py-6 justify-center">
              <span className="w-2 h-2 rounded-full bg-[var(--fit-good)] animate-pulse" />
              {t("models:search.searchingFor", { query })}
            </div>
          ) : error ? (
            <p className="text-[var(--fit-tight)] text-sm py-3">{error}</p>
          ) : results.length > 0 ? (
            <>
              <div className="flex items-center gap-3 mb-3 flex-wrap">
                <span className="eyebrow">
                  {t("models:search.resultsFor", { count: results.length, term: searchedTerm })}
                </span>
                <span className="h-px flex-1 bg-white/10" />
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="mono text-[11px] rounded-lg border border-[var(--line)] bg-[var(--canvas)] text-[var(--ink-dim)] px-2 py-1 focus:outline-none focus:border-[var(--fit-good)] transition-colors cursor-pointer"
                >
                  {SORT_KEYS.map((key) => (
                    <option key={key} value={key}>
                      {t(`models:search.sort.${key}`)}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => setCollapsed((c) => !c)}
                  className="mono text-[11px] text-[var(--ink-dim)] hover:text-[var(--fit-good)] transition-colors"
                >
                  {collapsed ? t("models:search.show") : t("models:search.collapse")}
                </button>
                <button
                  onClick={clear}
                  className="mono text-[11px] text-[var(--ink-dim)] hover:text-[var(--ink)] transition-colors"
                >
                  {t("common:actions.clear")}
                </button>
              </div>
              {!collapsed && (
                <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 max-h-[560px] overflow-y-auto custom-scroll pr-1">
                  {sortedResults.map((model) => (
                    <ExploreModelCard
                      key={`hf-${model.link}`}
                      model={model}
                      range={range}
                      onDownload={onDownload}
                      onInfo={onInfo}
                      installed={isInstalled ? isInstalled(model) : false}
                    />
                  ))}
                </div>
              )}
            </>
          ) : (
            <p className="text-[var(--ink-dim)] text-sm py-3">
              {t("models:search.noMatch", { term: searchedTerm })}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

HuggingFaceSearchPanel.propTypes = {
  range: PropTypes.shape({ min: PropTypes.number, max: PropTypes.number }),
  onDownload: PropTypes.func,
  onInfo: PropTypes.func,
  isInstalled: PropTypes.func,
  handoff: PropTypes.shape({ term: PropTypes.string, seq: PropTypes.number }),
};
