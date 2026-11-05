import React, { useId } from "react";
import PropTypes from "prop-types";
import { HelpCircle } from "lucide-react";
import Tooltip from "./Tooltip";

/**
 * The `?` beside a popup field (#585): a button Tab reaches, whose help text
 * shows on hover and on keyboard focus and describes it for screen readers.
 */
export default function HelpTip({ label, text, side = "right", width = "w-64" }) {
  const id = useId();
  return (
    <Tooltip content={text} side={side} width={width} id={id}>
      <button
        type="button"
        aria-label={label}
        aria-describedby={id}
        className="inline-flex rounded-full text-gray-400 hover:text-emerald-400 focus:text-emerald-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60 transition-colors cursor-help"
      >
        <HelpCircle className="w-4 h-4" aria-hidden />
      </button>
    </Tooltip>
  );
}

HelpTip.propTypes = {
  label: PropTypes.string.isRequired,
  text: PropTypes.string.isRequired,
  side: PropTypes.string,
  width: PropTypes.string,
};
