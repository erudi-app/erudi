import React, { useEffect, useRef, useState } from "react";
import PropTypes from "prop-types";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, HelpCircle, SlidersHorizontal } from "lucide-react";
import { useTranslation } from "react-i18next";
import Tooltip from "./Tooltip";
import ToggleSwitch from "./ToggleSwitch";
import { formatNumber } from "../i18n/format";

// Temperature / top-p read-outs: two decimals in the active locale.
const TWO_DECIMALS = { minimumFractionDigits: 2, maximumFractionDigits: 2 };

export default function HeaderBar({
  initialTemperature = 0.2,
  initialTopP = 0.2,
  initialMaxTokens = 1024,
  // Ceiling of the max-tokens field (#388): min(model context window, engine
  // window) resolved by the backend per model; the API's own upper bound
  // when the consumer has no model to derive it from.
  maxTokensCap = 32768,
  onApply,
  // Optional live callback (#218): when provided, every slider/token edit is
  // pushed to the parent immediately, so the displayed value is the value used
  // at send time. onApply stays intact for consumers (ConversationPage) that
  // deliberately commit-and-persist on an explicit Apply instead.
  onLiveChange,
  onCustomizePrompt,
  disabled = false,
  models = [],
  currentModel = "",
  onModelChange,
  // Attention state for the model picker (#225): when the open conversation's
  // model was deleted, the picker turns red and a short prompt sits beside it,
  // steering the user to pick a model before the composer will unblock.
  pickerAttention = false,
  pickerAttentionMessage = "",
  // Per-conversation web-search toggle (#310). Hidden by default so the
  // Arena (which shares this bar but has no conversation row — its turns
  // follow the GLOBAL setting backend-side) keeps an unchanged panel;
  // ConversationPage opts in and persists flips through onWebSearchChange.
  showWebSearch = false,
  initialWebSearch = false,
  onWebSearchChange,
  // Positive-knowledge gate (#570): true when the conversation's current
  // model is known unable to execute tools, so the `web_search` tool never
  // joins the turn no matter this toggle's state. The toggle stays visible
  // (its stored value is not hidden) but is disabled with an explanatory
  // tooltip instead of silently doing nothing when flipped on. The caller
  // (ConversationPage) computes this from the models list it already loads;
  // this component stays decoupled from that lookup.
  webSearchDisabled = false,
  // The model's publisher gives no sampling recommendation (#388,
  // `sampling_defaults.source === "none"`): a discreet line under the sliders
  // says the neutral defaults apply. Nothing is shown when one exists.
  noPublisherRecommendation = false,
}) {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const [temperature, setTemperature] = useState(initialTemperature);
  const [topP, setTopP] = useState(initialTopP);
  const [maxTokens, setMaxTokens] = useState(initialMaxTokens);
  const [webSearch, setWebSearch] = useState(initialWebSearch);

  // Sync internal state with props when they change
  useEffect(() => {
    setTemperature(initialTemperature);
  }, [initialTemperature]);

  useEffect(() => {
    setWebSearch(initialWebSearch);
  }, [initialWebSearch]);

  useEffect(() => {
    setTopP(initialTopP);
  }, [initialTopP]);

  useEffect(() => {
    setMaxTokens(initialMaxTokens);
  }, [initialMaxTokens]);

  const rootRef = useRef(null);
  const dropdownRef = useRef(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [tier, setTier] = useState("lg");

  useEffect(() => {
    if (!rootRef.current) {
      return;
    }
    const el = rootRef.current;

    const computeTier = (w) => {
      if (w < 360) {
        return "xs";
      }
      if (w < 520) {
        return "sm";
      }
      if (w < 720) {
        return "md";
      }
      return "lg";
    };

    const ro = new ResizeObserver(([entry]) => {
      const w = entry?.contentRect?.width ?? el.offsetWidth ?? 9999;
      setTier((prev) => {
        const next = computeTier(w);
        return prev === next ? prev : next;
      });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const isXs = tier === "xs";
  const isSm = tier === "sm" || tier === "xs";
  const isMd = tier === "md";
  const isNarrow = isSm || isXs;

  const handleApply = () => {
    onApply?.({ temperature, topP, maxTokens });
    setIsOpen(false);
  };

  const TooltipIcon = ({ id, side = "right" }) => {
    const text =
      id === "temperature"
        ? t("chat:header.tooltips.temperature")
        : id === "top-p"
          ? t("chat:header.tooltips.topP")
          : id === "prompt"
            ? t("chat:header.tooltips.prompt")
            : id === "web-search"
              ? t("chat:header.tooltips.webSearch")
              : "";
    const widthClass = isXs ? "w-40" : isSm ? "w-52" : "w-64";
    const iconSize = isXs ? "w-3 h-3" : isSm ? "w-3.5 h-3.5" : "w-4 h-4";
    return (
      <Tooltip content={text} side={side} width={widthClass}>
        <HelpCircle
          className={`${iconSize} text-gray-400 hover:text-emerald-400 transition-colors cursor-help`}
        />
      </Tooltip>
    );
  };

  const sliderBg = (value, max = 1) => {
    const pct = Math.round((value / max) * 100);
    return {
      background: `linear-gradient(to right, #25C08A 0%, #1EAB78 ${pct}%, rgba(255,255,255,0.06) ${pct}%, rgba(255,255,255,0.06) 100%)`,
    };
  };

  // Size-aware utility fragments
  const pad = isXs ? "p-3" : isSm ? "p-4" : "p-5";
  const titleText = isXs ? "text-[0.95rem]" : isSm ? "text-[1.02rem]" : "text-[1.15rem]";
  const pillPx = isXs ? "px-2.5" : isSm ? "px-3" : "px-3.5";
  const pillPy = isXs ? "py-1" : "py-1.5";
  const pillText = isXs ? "text-xs" : isSm ? "text-[0.8rem]" : "text-sm";
  const selectPadRight = isXs ? "pr-4" : "pr-5";
  const toggleSize = isXs
    ? "w-8 h-8 rounded-lg"
    : isSm
      ? "w-8 h-8 rounded-lg"
      : "w-9 h-9 rounded-xl";
  const labelText = isXs ? "text-[0.65rem]" : isSm ? "text-[0.7rem]" : "text-[0.72rem]";
  const statText = isXs ? "text-[10px]" : "text-[11px]";
  const numberWidth = isXs ? "w-20" : isSm ? "w-24" : "w-28";
  const primaryBtn = isXs
    ? "px-4 py-1.5 text-[0.8rem]"
    : isSm
      ? "px-4 py-1.5 text-[0.85rem]"
      : "px-5 py-2 text-[0.9rem]";
  const secondaryBtn = isXs
    ? "px-3.5 py-1.5 text-[0.78rem]"
    : isSm
      ? "px-4 py-1.5 text-[0.8rem]"
      : "px-4 py-2 text-sm";

  // Stack buttons on narrow widths
  const actionsLayout = isNarrow ? "flex-col items-stretch" : "flex-row items-center";

  // One-column layout when narrow; two columns otherwise
  const gridColsClass = isNarrow ? "grid-cols-1" : "md:grid-cols-2";

  return (
    <div
      ref={rootRef}
      className={[
        "hb-scope relative w-full rounded-[26px]",
        "border border-white/10",
        "bg-[rgba(22,40,36,0.45)] backdrop-blur-[18px] saturate-[1.4]",
        "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)]",
        disabled ? "opacity-50 pointer-events-none select-none" : "",
        isXs ? "hb-xs" : isSm ? "hb-sm" : isMd ? "hb-md" : "hb-lg",
      ].join(" ")}
    >
      <style>{`
        .hb-scope input.hb-range { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 999px; outline: none; }
        .hb-scope input.hb-range::-webkit-slider-thumb {
          -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; border: 0; cursor: pointer;
          background: radial-gradient(circle at 30% 30%, #ffffff, #d9e4dd 60%, #b7c6c0 100%);
          box-shadow: 0 2px 6px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.4), inset 0 1px 2px rgba(255,255,255,0.7);
          transition: transform .25s ease, box-shadow .25s ease;
        }
        /* Compact thumb sizes */
        .hb-scope.hb-sm input.hb-range::-webkit-slider-thumb { width: 16px; height: 16px; }
        .hb-scope.hb-xs input.hb-range::-webkit-slider-thumb { width: 14px; height: 14px; }
        .hb-scope input.hb-range:hover::-webkit-slider-thumb { transform: scale(1.07); }
        .hb-scope input.hb-range:active::-webkit-slider-thumb { transform: scale(.9); }
        .hb-scope input.hb-range:focus-visible::-webkit-slider-thumb {
          box-shadow: 0 0 0 4px rgba(37,192,138,0.35), 0 2px 6px rgba(0,0,0,0.55), inset 0 1px 2px rgba(255,255,255,0.8);
        }
        /* Firefox */
        .hb-scope input.hb-range::-moz-range-track { height: 6px; background: rgba(255,255,255,0.06); border-radius: 999px; }
        .hb-scope input.hb-range::-moz-range-thumb {
          width: 18px; height: 18px; border-radius: 50%; border: 0; cursor: pointer;
          background: radial-gradient(circle at 30% 30%, #ffffff, #d9e4dd 60%, #b7c6c0 100%);
          box-shadow: 0 2px 6px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.4), inset 0 1px 2px rgba(255,255,255,0.7);
        }
        .hb-scope.hb-sm input.hb-range::-moz-range-thumb { width: 16px; height: 16px; }
        .hb-scope.hb-xs input.hb-range::-moz-range-thumb { width: 14px; height: 14px; }
        .hb-scope input.hb-range:focus-visible::-moz-range-thumb {
          box-shadow: 0 0 0 4px rgba(37,192,138,0.35), 0 2px 6px rgba(0,0,0,0.55), inset 0 1px 2px rgba(255,255,255,0.8);
        }
      `}</style>

      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none rounded-[26px] mix-blend-overlay"
        style={{
          background: "linear-gradient(to bottom, rgba(255,255,255,0.18), rgba(255,255,255,0) 40%)",
        }}
      />

      <div className={`relative z-10 ${pad}`}>
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 flex-wrap min-w-0">
            <h3
              className={`${titleText} font-semibold tracking-tight text-[#F2F7F4] truncate`}
              title={t("chat:header.chatWith")}
            >
              {t("chat:header.chatWith")}
            </h3>

            <div
              ref={dropdownRef}
              role="button"
              aria-label={t("chat:header.selectModel")}
              className={[
                "inline-flex items-center rounded-lg relative",
                pillPx,
                pillPy,
                pillText,
                "border transition",
                pickerAttention
                  ? "bg-red-500/10 hover:bg-red-500/15 border-red-500/70 hover:border-red-500"
                  : "bg-white/5 hover:bg-white/10 border-white/10 hover:border-white/20",
                "backdrop-blur-sm text-gray-100",
                "max-w-[100%] cursor-pointer",
              ].join(" ")}
              onClick={() => !disabled && setIsDropdownOpen(!isDropdownOpen)}
            >
              <div
                className={[
                  "font-medium truncate",
                  selectPadRight,
                  isNarrow ? "max-w-[110px]" : "max-w-[150px]",
                ].join(" ")}
                title={currentModel}
              >
                {currentModel || t("chat:header.selectModelPlaceholder")}
              </div>
              <ChevronDown
                size={isXs ? 14 : 16}
                className={`opacity-70 shrink-0 transition-transform ${
                  isDropdownOpen ? "rotate-180" : ""
                }`}
              />

              {/* Custom Dropdown */}
              {isDropdownOpen && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-[#2a2a2a] border border-white/20 rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
                  {models.map((m) => (
                    <div
                      key={m.id ?? m.name}
                      className="px-3 py-2 hover:bg-white/10 cursor-pointer text-gray-100 border-b border-white/10 last:border-b-0"
                      onClick={(e) => {
                        e.stopPropagation();
                        onModelChange?.(m.name);
                        setIsDropdownOpen(false);
                      }}
                    >
                      {m.name}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {pickerAttention && pickerAttentionMessage && (
              <span role="alert" className="text-xs font-medium text-red-400 whitespace-nowrap">
                {pickerAttentionMessage}
              </span>
            )}
          </div>

          <button
            type="button"
            aria-label={t("chat:header.toggleSettings")}
            onClick={() => setIsOpen((v) => !v)}
            className={[
              "inline-flex items-center justify-center",
              toggleSize,
              "bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20",
              "text-gray-300 hover:text-emerald-400 transition",
              "shrink-0",
            ].join(" ")}
          >
            <SlidersHorizontal size={isXs ? 16 : 18} />
          </button>
        </div>

        <AnimatePresence initial={false}>
          {isOpen && (
            <motion.div
              key="controls"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ type: "tween", duration: 0.3 }}
              className="overflow-hidden"
            >
              <div className={`mt-6 grid gap-6 ${gridColsClass}`}>
                <div className="flex flex-col gap-6">
                  <div className="relative">
                    <div className="flex items-center gap-1.5 mb-1">
                      <span
                        className={`${labelText} uppercase tracking-wide font-semibold text-gray-300/80`}
                      >
                        {t("chat:header.creativity")}
                      </span>
                      <TooltipIcon id="temperature" side={isNarrow ? "bottom-right" : "right"} />
                      {/* Backend accepts 0-2; values above 1.0 get an amber tint
                          as a light "high creativity" cue (no redesign). */}
                      <span
                        className={`ml-auto ${statText} font-semibold px-2 py-0.5 rounded-md border ${
                          temperature > 1
                            ? "text-amber-200/90 bg-amber-500/10 border-amber-400/30"
                            : "text-emerald-200/90 bg-emerald-500/10 border-emerald-400/25"
                        }`}
                      >
                        {formatNumber(temperature, TWO_DECIMALS)}
                      </span>
                    </div>

                    <div className="relative pt-1">
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.01"
                        value={temperature}
                        onChange={(e) => {
                          const value = parseFloat(e.target.value);
                          setTemperature(value);
                          onLiveChange?.({ temperature: value, topP, maxTokens });
                        }}
                        className="hb-range w-full rounded-full bg-white/5 cursor-pointer"
                        style={sliderBg(temperature, 2)}
                      />
                    </div>
                  </div>

                  <div className="relative">
                    <div className="flex items-center gap-1.5 mb-1">
                      <span
                        className={`${labelText} uppercase tracking-wide font-semibold text-gray-300/80`}
                      >
                        {t("chat:header.diversity")}
                      </span>
                      <TooltipIcon id="top-p" side="right" />
                      <span
                        className={`ml-auto ${statText} font-semibold text-emerald-200/90 bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-400/25`}
                      >
                        {formatNumber(topP, TWO_DECIMALS)}
                      </span>
                    </div>

                    <div className="relative pt-1">
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={topP}
                        onChange={(e) => {
                          const value = parseFloat(e.target.value);
                          setTopP(value);
                          onLiveChange?.({ temperature, topP: value, maxTokens });
                        }}
                        className="hb-range w-full rounded-full bg-white/5 cursor-pointer"
                        style={sliderBg(topP)}
                      />
                    </div>
                  </div>
                  {noPublisherRecommendation && (
                    <p
                      data-testid="no-publisher-recommendation"
                      className="-mt-3 text-[11px] leading-snug text-gray-400/80"
                    >
                      {t("chat:header.noPublisherRecommendation")}
                    </p>
                  )}
                </div>

                <div className="flex flex-col justify-center gap-6">
                  <div>
                    {/* Labels row */}
                    <div className="grid grid-cols-2 items-start justify-items-start gap-x-6 gap-y-2 mb-2">
                      <div>
                        <span
                          className={`${labelText} uppercase tracking-wide font-semibold text-gray-300/80`}
                        >
                          {t("chat:header.maxTokens")}
                        </span>
                      </div>
                      {/* <div className="flex items-center gap-2">
                        <span
                          className={`${labelText} uppercase tracking-wide font-semibold text-gray-300/80`}
                        >
                          Low-Memory
                        </span>
                        <TooltipIcon id="quantize" side="bottom-left" />
                      </div> */}

                      {/* Controls row */}
                      <div className="inline-flex items-center rounded-md bg-white/10 border border-white/20 shadow p-0 m-0">
                        <input
                          type="number"
                          min="1"
                          max={maxTokensCap}
                          value={maxTokens}
                          onChange={(e) => {
                            const value = parseInt(e.target.value || "0", 10);
                            setMaxTokens(value);
                            onLiveChange?.({ temperature, topP, maxTokens: value });
                          }}
                          className={`bg-transparent border-0 outline-none ${numberWidth} text-sm font-semibold text-gray-100 text-center appearance-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none`}
                        />
                      </div>

                      {/* <div>
                        <button
                          type="button"
                          onClick={() => setQuantize(!quantize)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            quantize
                              ? "bg-emerald-600 hover:bg-emerald-700"
                              : "bg-white/20 hover:bg-white/30"
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              quantize ? "translate-x-6" : "translate-x-1"
                            }`}
                          />
                        </button>
                      </div> */}
                    </div>
                    {showWebSearch && (
                      <div className="mt-4 flex items-center gap-3">
                        <span
                          className={`${labelText} uppercase tracking-wide font-semibold text-gray-300/80`}
                        >
                          {t("chat:header.webSearch")}
                        </span>
                        <TooltipIcon id="web-search" side={isNarrow ? "bottom-right" : "right"} />
                        <div className="ml-auto">
                          {webSearchDisabled ? (
                            <Tooltip
                              content={t("chat:header.tooltips.webSearchUnavailable")}
                              side={isNarrow ? "bottom-right" : "right"}
                              width={isXs ? "w-40" : isSm ? "w-52" : "w-64"}
                            >
                              <ToggleSwitch
                                checked={webSearch}
                                onChange={(next) => {
                                  setWebSearch(next);
                                  onWebSearchChange?.(next);
                                }}
                                label={t("chat:header.webSearchToggle")}
                                disabled
                              />
                            </Tooltip>
                          ) : (
                            <ToggleSwitch
                              checked={webSearch}
                              onChange={(next) => {
                                setWebSearch(next);
                                onWebSearchChange?.(next);
                              }}
                              label={t("chat:header.webSearchToggle")}
                            />
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className={`flex ${actionsLayout} gap-3 w-full`}>
                    <div
                      className={`flex ${
                        isNarrow ? "items-center gap-2" : "items-center gap-2"
                      } w-full`}
                    >
                      <button
                        type="button"
                        onClick={onCustomizePrompt}
                        className={[
                          "rounded-md font-semibold",
                          primaryBtn,
                          "bg-emerald-800 hover:bg-emerald-900 text-white",
                          "border border-white/20 shadow",
                          "transition active:scale-95",
                          isNarrow ? "flex-1" : "",
                        ].join(" ")}
                      >
                        {t("chat:header.customizePrompt")}
                      </button>
                      <div>
                        <TooltipIcon id="prompt" side="top-left" />
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={handleApply}
                      className={[
                        "rounded-lg font-semibold",
                        secondaryBtn,
                        "bg-white/10 hover:bg-white/15 text-gray-100",
                        "border border-white/20 backdrop-blur-sm shadow-sm",
                        "transition active:scale-95",
                        isNarrow ? "w-full" : "ml-auto",
                      ].join(" ")}
                    >
                      {t("chat:header.apply")}
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

HeaderBar.propTypes = {
  initialTemperature: PropTypes.number,
  initialTopP: PropTypes.number,
  initialMaxTokens: PropTypes.number,
  maxTokensCap: PropTypes.number,
  onApply: PropTypes.func.isRequired,
  onLiveChange: PropTypes.func,
  onCustomizePrompt: PropTypes.func.isRequired,
  showWebSearch: PropTypes.bool,
  initialWebSearch: PropTypes.bool,
  onWebSearchChange: PropTypes.func,
  webSearchDisabled: PropTypes.bool,
  disabled: PropTypes.bool,
  models: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      name: PropTypes.string.isRequired,
    })
  ),
  currentModel: PropTypes.string,
  onModelChange: PropTypes.func,
  pickerAttention: PropTypes.bool,
  pickerAttentionMessage: PropTypes.string,
  noPublisherRecommendation: PropTypes.bool,
};
