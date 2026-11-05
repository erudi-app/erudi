// The /context panel's figures (#608), from `GET /conversations/{id}/context`.
//
// The categories, in the legend's and the grid's order: what the next request
// would carry (system prompt, tools, messages, the previous summary), then the
// free space left below the compaction threshold, then the auto-compaction
// reserve above it. While the request fits, the six add up to the working
// window W; past it, free and reserve are 0 and what is used fills the grid.

export const CONTEXT_CATEGORIES = Object.freeze([
  "system_prompt",
  "tools",
  "messages",
  "summary",
  "free",
  "autocompaction_buffer",
]);

const USED = ["system_prompt", "tools", "messages", "summary"];

export const CONTEXT_GRID_BLOCKS = 100;

const tokensOf = (categories, key) => {
  const n = Number(categories?.[key]);
  return Number.isFinite(n) && n > 0 ? n : 0;
};

/**
 * The grid's blocks for `rows` ({key, tokens}), out of what the rows hold
 * together: each row gets its share of `CONTEXT_GRID_BLOCKS` (largest
 * remainders, handed out until the blocks add up exactly), and a row that
 * holds tokens never rounds down to no block at all -- it takes one from the
 * largest. With the backend's payloads the rows add up to the window (or to
 * what is used, past it); a payload that does not still fills the grid.
 * Returns the block keys in row order, none when the rows hold nothing.
 */
function allocateBlocks(rows) {
  const total = rows.reduce((sum, r) => sum + r.tokens, 0);
  if (total <= 0) return [];
  const exact = rows.map((r) => (r.tokens / total) * CONTEXT_GRID_BLOCKS);
  const counts = exact.map(Math.floor);
  let left = CONTEXT_GRID_BLOCKS - counts.reduce((a, b) => a + b, 0);
  const byRemainder = exact
    .map((value, i) => ({ i, rest: value - counts[i] }))
    .sort((a, b) => b.rest - a.rest);
  for (let k = 0; left > 0; k = (k + 1) % byRemainder.length) {
    counts[byRemainder[k].i] += 1;
    left -= 1;
  }
  rows.forEach((r, i) => {
    if (r.tokens > 0 && counts[i] === 0) {
      const largest = counts.indexOf(Math.max(...counts));
      if (counts[largest] > 1) {
        counts[largest] -= 1;
        counts[i] = 1;
      }
    }
  });
  return rows.flatMap((r, i) => Array(counts[i]).fill(r.key));
}

/**
 * What the panel shows of `usage`: the window W (`window.working`, null when
 * unknown), the tokens used, one row per category ({key, tokens, share of W
 * or null}; the summary only when there is one), the grid's blocks (none
 * without W), and whether the conversation is past its compaction threshold
 * (no free space left while something is used).
 */
export function contextUsageView(usage) {
  const categories = usage?.categories ?? {};
  const working = Number(usage?.window?.working);
  const window = Number.isFinite(working) && working > 0 ? working : null;
  const used = USED.reduce((sum, key) => sum + tokensOf(categories, key), 0);
  const rows = CONTEXT_CATEGORIES.filter(
    (key) => key !== "summary" || tokensOf(categories, key) > 0
  ).map((key) => {
    const tokens = tokensOf(categories, key);
    return { key, tokens, share: window ? tokens / window : null };
  });
  const blocks = window ? allocateBlocks(rows) : [];
  return {
    window,
    used,
    rows,
    blocks,
    pastThreshold: window !== null && used > 0 && tokensOf(categories, "free") === 0,
  };
}
