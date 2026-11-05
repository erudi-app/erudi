import { describe, it, expect } from "vitest";
import { placeTooltip, TOOLTIP_GAP, TOOLTIP_MARGIN } from "./tooltipPlacement";

// Where a help tooltip goes (#585): inside the popup that holds its `?`,
// never clipped, below the trigger when it fits, above otherwise.

const rect = (left, top, width, height) => ({
  left,
  top,
  width,
  height,
  right: left + width,
  bottom: top + height,
});

// A popup 600 px wide, 400 px high, at (400, 200).
const POPUP = rect(400, 200, 600, 400);

describe("placeTooltip", () => {
  it("goes below the trigger, centred on it, when it fits", () => {
    const trigger = rect(600, 250, 16, 16);
    const place = placeTooltip(trigger, { width: 200, height: 60 }, POPUP);
    expect(place.side).toBe("bottom");
    expect(place.top).toBe(266 + TOOLTIP_GAP);
    expect(place.left).toBe(608 - 100);
  });

  it("flips above when the space below runs out", () => {
    const trigger = rect(600, 560, 16, 16);
    const place = placeTooltip(trigger, { width: 200, height: 60 }, POPUP);
    expect(place.side).toBe("top");
    expect(place.top).toBe(560 - TOOLTIP_GAP - 60);
  });

  it("goes beside the trigger, inside the popup, when a short popup has no room above or below", () => {
    // /effort: a 190 px high popup, a 125 px high tooltip.
    const short = rect(464, 354, 512, 190);
    const trigger = rect(622, 452, 16, 16);
    const place = placeTooltip(trigger, { width: 256, height: 125 }, short);
    expect(place.side).toBe("right");
    expect(place.left).toBe(638 + TOOLTIP_GAP);
    expect(place.top).toBeGreaterThanOrEqual(short.top + TOOLTIP_MARGIN);
    expect(place.top + 125).toBeLessThanOrEqual(short.bottom - TOOLTIP_MARGIN);
  });

  it("goes to the left when the right has no room", () => {
    const short = rect(464, 354, 512, 190);
    const trigger = rect(900, 452, 16, 16);
    const place = placeTooltip(trigger, { width: 256, height: 125 }, short);
    expect(place.side).toBe("left");
    expect(place.left + 256).toBe(900 - TOOLTIP_GAP);
  });

  it("falls back to the window, below or above where there is more room, when the popup holds it nowhere", () => {
    const tiny = rect(400, 300, 300, 100);
    const windowBounds = rect(0, 0, 1440, 900);
    const place = placeTooltip(
      rect(500, 340, 16, 16),
      { width: 256, height: 125 },
      tiny,
      windowBounds
    );
    expect(place.side).toBe("bottom");
    expect(place.top).toBe(356 + TOOLTIP_GAP);
    // Still kept within the popup's width.
    expect(place.left).toBeGreaterThanOrEqual(tiny.left + TOOLTIP_MARGIN);
  });

  it("stays inside the popup's right edge for a field in the right column", () => {
    const trigger = rect(980, 250, 16, 16);
    const place = placeTooltip(trigger, { width: 256, height: 60 }, POPUP);
    expect(place.left + 256).toBe(POPUP.right - TOOLTIP_MARGIN);
  });

  it("stays inside the popup's left edge", () => {
    const trigger = rect(405, 250, 16, 16);
    const place = placeTooltip(trigger, { width: 256, height: 60 }, POPUP);
    expect(place.left).toBe(POPUP.left + TOOLTIP_MARGIN);
  });

  it("caps its width to the popup on a narrow window", () => {
    const narrow = rect(10, 100, 220, 400);
    const place = placeTooltip(rect(100, 150, 16, 16), { width: 256, height: 60 }, narrow);
    expect(place.maxWidth).toBe(220 - 2 * TOOLTIP_MARGIN);
    expect(place.left).toBeGreaterThanOrEqual(narrow.left + TOOLTIP_MARGIN);
  });
});
