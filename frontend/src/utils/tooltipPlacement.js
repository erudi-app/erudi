// Where a help tooltip goes (#585). The tooltip is drawn with fixed
// positioning above everything, so no container clips it; this picks a spot
// inside `bounds` (the popup holding the `?`) where it is whole and readable.

/** Space between the trigger and the tooltip, in px. */
export const TOOLTIP_GAP = 8;
/** Space kept between the tooltip and the edge of its bounds, in px. */
export const TOOLTIP_MARGIN = 8;

const clamp = (value, min, max) => Math.min(Math.max(value, min), Math.max(min, max));

/**
 * `trigger`, `bounds` and `outer` are DOMRect-like ({left, top, right,
 * bottom, width, height}); `size` is the tooltip's {width, height}. Returns
 * {top, left, maxWidth, side}, the width capped to the bounds.
 *
 * The first spot that holds the whole tooltip inside `bounds` wins: below the
 * trigger, above it (both centred on it and shifted to stay inside), to its
 * right, to its left (both vertically centred on it, shifted to stay
 * inside). When `bounds` holds it nowhere (a popup shorter than the text),
 * it goes below or above, whichever has room in `outer` (the window), still
 * kept within the popup's width.
 */
export function placeTooltip(trigger, size, bounds, outer = bounds) {
  const maxWidth = Math.max(0, bounds.right - bounds.left - 2 * TOOLTIP_MARGIN);
  const width = Math.min(size.width, maxWidth);
  const height = size.height;

  const centredLeft = clamp(
    trigger.left + trigger.width / 2 - width / 2,
    bounds.left + TOOLTIP_MARGIN,
    bounds.right - TOOLTIP_MARGIN - width
  );
  const below = trigger.bottom + TOOLTIP_GAP;
  const above = trigger.top - TOOLTIP_GAP - height;
  const fitsVertically = (top) =>
    top >= bounds.top + TOOLTIP_MARGIN && top + height <= bounds.bottom - TOOLTIP_MARGIN;

  if (fitsVertically(below)) return { top: below, left: centredLeft, maxWidth, side: "bottom" };
  if (fitsVertically(above)) return { top: above, left: centredLeft, maxWidth, side: "top" };

  const besideTop = clamp(
    trigger.top + trigger.height / 2 - height / 2,
    bounds.top + TOOLTIP_MARGIN,
    bounds.bottom - TOOLTIP_MARGIN - height
  );
  const tallEnough = height <= bounds.bottom - bounds.top - 2 * TOOLTIP_MARGIN;
  const right = trigger.right + TOOLTIP_GAP;
  if (tallEnough && right + width <= bounds.right - TOOLTIP_MARGIN) {
    return { top: besideTop, left: right, maxWidth, side: "right" };
  }
  const left = trigger.left - TOOLTIP_GAP - width;
  if (tallEnough && left >= bounds.left + TOOLTIP_MARGIN) {
    return { top: besideTop, left, maxWidth, side: "left" };
  }

  const roomBelow = outer.bottom - TOOLTIP_MARGIN - below;
  const roomAbove = trigger.top - TOOLTIP_GAP - (outer.top + TOOLTIP_MARGIN);
  const side = roomBelow >= height || roomBelow >= roomAbove ? "bottom" : "top";
  return { top: side === "bottom" ? below : above, left: centredLeft, maxWidth, side };
}
