import React from "react";
import PropTypes from "prop-types";

// Reusable Tooltip component
export default function Tooltip({ children, content, side = "right", width = "w-64" }) {
  const positionClasses =
    {
      top: "bottom-full left-1/2 -translate-x-1/2 mb-2",
      bottom: "top-full left-1/2 -translate-x-1/2 mt-2",
      left: "right-full top-1/2 -translate-y-1/2 mr-2",
      right: "left-full top-1/2 -translate-y-1/2 ml-2",
      "top-left": "bottom-full right-0 mb-2",
      "top-right": "bottom-full left-0 mb-2",
      "bottom-left": "top-full right-0 mt-2",
      "bottom-right": "top-full left-0 mt-2",
    }[side] || "left-full top-1/2 -translate-y-1/2 ml-2";

  const arrowPosition =
    {
      right: "left-0 -translate-x-1/2 top-1/2 -translate-y-1/2",
      left: "right-0 translate-x-1/2 top-1/2 -translate-y-1/2",
      top: "left-1/2 -translate-x-1/2 bottom-0 translate-y-1/2",
      bottom: "left-1/2 -translate-x-1/2 top-0 -translate-y-1/2",
      "top-left": "right-0 bottom-0 translate-x-1/2 translate-y-1/2",
      "top-right": "left-0 bottom-0 -translate-x-1/2 translate-y-1/2",
      "bottom-left": "right-0 top-0 translate-x-1/2 -translate-y-1/2",
      "bottom-right": "left-0 top-0 -translate-x-1/2 -translate-y-1/2",
    }[side] || "left-0 -translate-x-1/2 top-1/2 -translate-y-1/2";

  return (
    <span className="relative group inline-block align-middle overflow-visible">
      {children}
      <span
        className={`pointer-events-none absolute ${positionClasses} ${width} opacity-0 group-hover:opacity-100 scale-95 group-hover:scale-100 transition-all duration-300 ease-out z-[999999] overflow-visible`}
      >
        <span className="relative block px-4 py-3 text-sm text-gray-200 bg-gradient-to-br from-[#1a1a1a] via-[#2a2a2a] to-[#1a1a1a] rounded-xl shadow-2xl border border-emerald-500/20 backdrop-blur-sm font-normal">
          {content}
          <span className={`absolute w-3 h-3 ${arrowPosition} z-[-1]`}>
            <span className="block w-full h-full rotate-45 bg-[#1a1a1a] border border-emerald-500/20" />
          </span>
        </span>
      </span>
    </span>
  );
}
Tooltip.propTypes = {
  text: PropTypes.string.isRequired,
  children: PropTypes.node.isRequired,
};
