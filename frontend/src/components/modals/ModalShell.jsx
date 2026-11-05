import React, { useEffect, useId, useRef } from "react";
import PropTypes from "prop-types";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { useTranslation } from "react-i18next";

const FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

/**
 * The shell the command popups share (#585), in the app's glass modal style.
 * When it opens it takes the focus (its first field), keeps Tab and
 * Shift+Tab inside, closes on Escape, and gives the focus back to the element
 * that held it before. A popup never opens another one on top of itself.
 */
export default function ModalShell({
  isOpen,
  onClose,
  title,
  subtitle,
  children,
  width = "max-w-lg",
}) {
  const { t } = useTranslation();
  const titleId = useId();
  const subtitleId = useId();
  const panelRef = useRef(null);
  const closeRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return undefined;
    const previous = document.activeElement;
    const panel = panelRef.current;
    // The first field, never the close button nor a `?` help button (its
    // tooltip would open over the field it explains).
    const first = Array.from(panel?.querySelectorAll(FOCUSABLE) ?? []).find(
      (el) => el !== closeRef.current && !el.hasAttribute("data-help-tip")
    );
    (first ?? closeRef.current)?.focus();
    return () => {
      // The element that opened the popup may be gone (a re-rendered list).
      if (previous?.isConnected) previous.focus();
    };
  }, [isOpen]);

  // The keys are heard on the document while the popup is open: a click on a
  // heading, a label or the padding leaves the focus on the panel or on the
  // page body, and Escape and the Tab trap must still work from there.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    if (!isOpen) return undefined;
    const handleKeyDown = (e) => {
      // An input method composing (Chinese, Japanese): its Escape cancels the
      // composition, never the popup.
      if (e.defaultPrevented || e.isComposing || e.keyCode === 229) return;
      if (e.key === "Escape") {
        e.preventDefault();
        onCloseRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const panel = panelRef.current;
      const focusables = Array.from(panel?.querySelectorAll(FOCUSABLE) ?? []);
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement;
      // Focus on the panel itself (after a click) counts as outside its fields.
      const inside = panel.contains(active) && active !== panel;
      if (e.shiftKey && (active === first || !inside)) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || !inside)) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[9999] p-4"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) onClose();
          }}
        >
          <motion.div
            ref={panelRef}
            role="dialog"
            aria-modal
            aria-labelledby={titleId}
            aria-describedby={subtitle ? subtitleId : undefined}
            // A click on a spot with no field focuses the panel, so the keys
            // stay inside the popup.
            tabIndex={-1}
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className={`relative w-full outline-none ${width}`}
          >
            <div
              className={[
                "relative w-full rounded-[26px]",
                "border border-white/10",
                "bg-[rgba(22,40,36,0.85)] backdrop-blur-[18px] saturate-[1.4]",
                "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)]",
              ].join(" ")}
            >
              <div
                aria-hidden
                className="absolute inset-0 pointer-events-none rounded-[26px] mix-blend-overlay"
                style={{
                  background:
                    "linear-gradient(to bottom, rgba(255,255,255,0.18), rgba(255,255,255,0) 40%)",
                }}
              />
              <div className="relative z-10 p-6 max-h-[85vh] overflow-y-auto">
                <div className="flex items-start justify-between gap-4 mb-5">
                  <div className="min-w-0">
                    <h2
                      id={titleId}
                      className="text-xl font-semibold tracking-tight text-[#F2F7F4]"
                    >
                      {title}
                    </h2>
                    {subtitle && (
                      <p id={subtitleId} className="mt-1 text-sm text-gray-300/80">
                        {subtitle}
                      </p>
                    )}
                  </div>
                  <button
                    ref={closeRef}
                    type="button"
                    onClick={onClose}
                    aria-label={t("common:actions.close")}
                    className="inline-flex shrink-0 items-center justify-center w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 text-gray-300 hover:text-gray-100 transition"
                  >
                    <X className="w-4 h-4" aria-hidden />
                  </button>
                </div>
                {children}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

ModalShell.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  title: PropTypes.string.isRequired,
  subtitle: PropTypes.string,
  children: PropTypes.node,
  width: PropTypes.string,
};
