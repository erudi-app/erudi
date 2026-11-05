import React, { useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import PropTypes from "prop-types";
import { HelpCircle } from "lucide-react";
import { placeTooltip, TOOLTIP_MARGIN } from "../utils/tooltipPlacement";

// The help text's width when the popup leaves room for it, in px.
const TOOLTIP_WIDTH = 256;

const windowBounds = () => ({
  left: 0,
  top: 0,
  right: window.innerWidth,
  bottom: window.innerHeight,
});

/**
 * The `?` beside a popup field (#585): a button Tab reaches, described by its
 * help text, which shows on hover and on keyboard focus.
 *
 * A closed tooltip is hidden from assistive technology, and Escape on the
 * button closes the open tooltip without closing the popup.
 *
 * The text is drawn on <body> with fixed positioning, so the popup's
 * scrolling body (or any other container) never clips it, and it is placed
 * inside the popup holding the `?` (the window outside a popup): below the
 * button, above it, to its right or to its left, the first that holds it
 * whole, narrowed to the popup's width; the window takes over only when the
 * popup holds it nowhere (`placeTooltip`).
 */
export default function HelpTip({ label, text }) {
  const id = useId();
  const triggerRef = useRef(null);
  const tipRef = useRef(null);
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState(null);

  useLayoutEffect(() => {
    if (!open) return undefined;
    const update = () => {
      const trigger = triggerRef.current;
      const tip = tipRef.current;
      if (!trigger || !tip) return;
      const dialog = trigger.closest('[role="dialog"]');
      const bounds = dialog ? dialog.getBoundingClientRect() : windowBounds();
      // Measure the height at the width the tooltip will have.
      const width = Math.min(
        TOOLTIP_WIDTH,
        Math.max(0, bounds.right - bounds.left - 2 * TOOLTIP_MARGIN)
      );
      tip.style.width = `${width}px`;
      const next = placeTooltip(
        trigger.getBoundingClientRect(),
        { width, height: tip.offsetHeight },
        bounds,
        windowBounds()
      );
      setPlace({ ...next, width });
    };
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [open]);

  const show = () => setOpen(true);
  const hide = () => setOpen(false);
  const shown = open && place !== null;
  // Escape dismisses the open tooltip alone (WCAG 1.4.13); a second Escape
  // reaches the popup.
  const handleKeyDown = (e) => {
    if (e.key === "Escape" && open) {
      e.preventDefault();
      e.stopPropagation();
      hide();
    }
  };

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-label={label}
        aria-describedby={id}
        // The popup shell skips it when it places the first focus: focusing
        // it would open its tooltip over the field it explains.
        data-help-tip
        onFocus={show}
        onBlur={hide}
        onMouseEnter={show}
        onMouseLeave={hide}
        onKeyDown={handleKeyDown}
        className="inline-flex align-middle rounded-full text-gray-400 hover:text-emerald-400 focus:text-emerald-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60 transition-colors cursor-help"
      >
        <HelpCircle className="w-4 h-4" aria-hidden />
      </button>
      {createPortal(
        <div
          ref={tipRef}
          data-help-tooltip
          data-open={shown ? "true" : "false"}
          style={{
            position: "fixed",
            // Closed, it is out of the accessibility tree; the button's
            // aria-describedby still reads its text.
            visibility: shown ? "visible" : "hidden",
            top: place?.top ?? 0,
            left: place?.left ?? 0,
            width: place?.width ?? TOOLTIP_WIDTH,
          }}
          className={`pointer-events-none z-[10000] transition-opacity duration-200 ${
            shown ? "opacity-100" : "opacity-0"
          }`}
        >
          <div
            id={id}
            role="tooltip"
            className="px-4 py-3 text-sm text-gray-200 bg-gradient-to-br from-[#1a1a1a] via-[#2a2a2a] to-[#1a1a1a] rounded-xl shadow-2xl border border-emerald-500/20 font-normal normal-case tracking-normal"
          >
            {text}
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

HelpTip.propTypes = {
  label: PropTypes.string.isRequired,
  text: PropTypes.string.isRequired,
};
