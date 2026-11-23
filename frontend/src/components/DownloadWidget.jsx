import React from "react";
import PropTypes from "prop-types";
import { motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, Check, ChevronDown, ChevronUp, Download, Loader2, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { formatFileSize, formatGigabytes, formatPercent } from "../i18n/format";
import { BYTES_PER_GB } from "../utils/hardwareFit";
import { DOWNLOAD_PHASES } from "../utils/downloadStatus";

/**
 * The download progress widget (#292): one glass panel anchored bottom-left,
 * next to the navigation rail, that follows the user across pages while a
 * model is being fetched.
 *
 * Purely presentational. `DownloadModalContext` owns the state machine (poll,
 * stall guard, cancel, auto-dismiss) and hands this component a phase plus the
 * readouts derived from `GET /llms/downloads/{id}/status`; this file only
 * decides what each phase looks like. Mount it inside an `<AnimatePresence>`
 * so the exit transition plays when the context drops it.
 */

const ACTIVE_PHASES = new Set(["queued", "downloading", "finalizing"]);
const DISMISSABLE_PHASES = new Set(["failed", "stalled"]);
const PERCENT_PHASES = new Set(["downloading", "finalizing", "completed"]);

// Per-phase tint. Emerald is the brand's "in progress / good"; failure and
// stall reuse the fit palette (rust / amber) so the widget speaks the same
// colour language as the catalog gauges.
const TONES = {
  active: {
    color: "var(--fit-good)",
    border: "border-emerald-200/20",
    chip: "bg-emerald-500/15 text-emerald-300",
    fill: "from-emerald-600 via-emerald-500 to-emerald-400",
    tint: "rgba(16,185,129,",
  },
  completed: {
    color: "var(--fit-good)",
    border: "border-emerald-300/35",
    chip: "bg-emerald-500/20 text-emerald-200",
    fill: "from-emerald-500 to-emerald-400",
    tint: "rgba(52,214,165,",
  },
  cancelled: {
    color: "var(--ink-dim)",
    border: "border-white/10",
    chip: "bg-white/5 text-[var(--ink-dim)]",
    fill: "",
    tint: "rgba(233,245,240,",
  },
  failed: {
    color: "var(--fit-heavy)",
    border: "border-red-400/30",
    chip: "bg-red-500/15 text-red-300",
    fill: "",
    tint: "rgba(207,125,114,",
  },
  stalled: {
    color: "var(--fit-tight)",
    border: "border-amber-400/30",
    chip: "bg-amber-500/15 text-amber-300",
    fill: "from-amber-500 to-amber-400",
    tint: "rgba(232,177,76,",
  },
};

const toneFor = (phase) => (ACTIVE_PHASES.has(phase) ? TONES.active : TONES[phase]);

// Time left in the largest two units that matter ("1d 1h", "1h 1m", "1m 35s",
// "40s"); `null` when the estimate has no value yet.
function formatTimeLeft(t, seconds) {
  if (!seconds || seconds <= 0) return null;
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  if (days > 0) return t("downloads:widget.duration.daysHours", { days, hours });
  if (hours > 0) return t("downloads:widget.duration.hoursMinutes", { hours, minutes });
  if (minutes > 0) return t("downloads:widget.duration.minutesSeconds", { minutes, seconds: secs });
  return t("downloads:widget.duration.seconds", { seconds: secs });
}

function IconButton({ label, onClick, danger = false, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className={[
        "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border transition",
        "bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20",
        danger ? "text-gray-300 hover:text-red-300" : "text-gray-300 hover:text-white",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

IconButton.propTypes = {
  label: PropTypes.string.isRequired,
  onClick: PropTypes.func.isRequired,
  danger: PropTypes.bool,
  children: PropTypes.node,
};

/** Small progress arc for the collapsed pill. */
function ProgressRing({ fraction, color }) {
  const radius = 9;
  const circumference = 2 * Math.PI * radius;
  return (
    <svg width={24} height={24} viewBox="0 0 24 24" aria-hidden="true" className="-rotate-90">
      <circle
        cx={12}
        cy={12}
        r={radius}
        fill="none"
        stroke="rgba(255,255,255,0.1)"
        strokeWidth={2.5}
      />
      <circle
        cx={12}
        cy={12}
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth={2.5}
        strokeDasharray={circumference}
        strokeDashoffset={circumference * (1 - fraction)}
        style={{ strokeLinecap: "round", transition: "stroke-dashoffset 300ms ease-out" }}
      />
    </svg>
  );
}

ProgressRing.propTypes = {
  fraction: PropTypes.number.isRequired,
  color: PropTypes.string.isRequired,
};

/**
 * The phase glyph: a spinner while waiting on the backend, the progress ring
 * (with a small download arrow inside) while bytes are flowing, otherwise a
 * verdict icon.
 */
function PhaseGlyph({ phase, fraction, color, chipClass }) {
  if (phase === "downloading") {
    return (
      <div
        className={`relative flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${chipClass}`}
      >
        <ProgressRing fraction={fraction} color={color} />
        <Download className="absolute h-3 w-3" />
      </div>
    );
  }
  let icon;
  if (phase === "queued" || phase === "finalizing") {
    icon = <Loader2 className="h-4 w-4 motion-safe:animate-spin" />;
  } else if (phase === "completed") {
    icon = <Check className="h-4 w-4" />;
  } else if (phase === "cancelled") {
    icon = <X className="h-4 w-4" />;
  } else {
    icon = <AlertTriangle className="h-4 w-4" />;
  }
  return (
    <div
      className={`relative flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${chipClass}`}
    >
      {icon}
    </div>
  );
}

PhaseGlyph.propTypes = {
  phase: PropTypes.string.isRequired,
  fraction: PropTypes.number.isRequired,
  color: PropTypes.string.isRequired,
  chipClass: PropTypes.string.isRequired,
};

// Always in GB, like the model card (which never switches to MB): the same
// artifact reads the same unit in the card and in the widget (#753).
const formatTransferred = (bytes) => formatGigabytes(bytes / BYTES_PER_GB);

export default function DownloadWidget({
  modelName,
  phase,
  progress,
  timeLeft,
  totalBytes,
  downloadedBytes,
  speedBytesPerSec,
  message,
  collapsed,
  onToggleCollapse,
  onCancel,
  onDismiss,
}) {
  const { t } = useTranslation();
  const reduceMotion = useReducedMotion();
  const tone = toneFor(phase);
  const active = ACTIVE_PHASES.has(phase);
  const pct = Math.min(100, Math.max(0, progress ?? 0));
  const fraction = pct / 100;
  const percentLabel = PERCENT_PHASES.has(phase) ? formatPercent(pct) : null;
  const phaseLabel = t(`downloads:widget.phase.${phase}`);
  const showBar = active || phase === "completed";
  const eta = phase === "downloading" ? formatTimeLeft(t, timeLeft) : null;
  const bytesLabel =
    active && totalBytes > 0 && downloadedBytes !== null && downloadedBytes !== undefined
      ? t("downloads:widget.bytes", {
          done: formatTransferred(Math.min(downloadedBytes, totalBytes)),
          total: formatTransferred(totalBytes),
        })
      : null;
  const speedLabel =
    phase === "downloading" && speedBytesPerSec > 0
      ? t("downloads:widget.speed", { rate: formatFileSize(speedBytesPerSec) })
      : null;

  // Entrance / exit: a short rise from below the anchor, or a plain fade when
  // the OS asks for reduced motion.
  const hidden = reduceMotion ? { opacity: 0 } : { opacity: 0, y: 12, scale: 0.98 };
  const shown = reduceMotion ? { opacity: 1 } : { opacity: 1, y: 0, scale: 1 };
  const rootTransition = reduceMotion
    ? { duration: 0 }
    : { type: "spring", stiffness: 380, damping: 32, mass: 0.8 };
  const swapTransition = { duration: reduceMotion ? 0 : 0.12 };

  // The status line is the single live region: rendered visibly in the panel
  // and visually hidden in the pill, so assistive tech keeps following the
  // phase whichever way the user parked the widget.
  const statusLine = (hiddenVisually) => (
    <p
      role="status"
      aria-live="polite"
      className={hiddenVisually ? "sr-only" : "truncate text-xs leading-snug text-[var(--ink-dim)]"}
    >
      {phaseLabel}
    </p>
  );

  return (
    <motion.section
      aria-label={t("downloads:widget.title")}
      data-phase={phase}
      initial={hidden}
      animate={shown}
      exit={hidden}
      transition={rootTransition}
      className={[
        // Right of the 56px navigation rail (#347) and above the connection
        // pill that lives at the bottom of the models sidebar (#303).
        "fixed bottom-14 left-[4.25rem] z-50",
        collapsed ? "w-auto" : "w-[min(22rem,calc(100vw_-_5.5rem))]",
        // No `relative` here: Tailwind emits it after `fixed` and it would win,
        // dropping the panel into the document flow under the app.
        "overflow-hidden rounded-2xl border",
        tone.border,
        "bg-[rgba(12,39,34,0.72)] backdrop-blur-[14px] saturate-[1.3]",
        "shadow-[0_10px_30px_-6px_rgba(0,0,0,0.5),0_2px_6px_-1px_rgba(0,0,0,0.45)]",
        "text-[var(--ink)]",
      ].join(" ")}
    >
      {/* Frost overlay, tinted by phase (same recipe as the composer). */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 rounded-2xl mix-blend-overlay"
        style={{
          background: `linear-gradient(180deg, ${tone.tint}0.14) 0%, ${tone.tint}0.06) 30%, ${tone.tint}0) 100%)`,
        }}
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 rounded-2xl"
        style={{ boxShadow: `inset 0 1px 0 ${tone.tint}0.16)` }}
      />

      {/* Collapse / expand swaps the body outright (keyed remount with a short
          fade-in). No exit choreography here: an exiting body would keep a
          frozen snapshot of the previous phase on screen. */}
      {collapsed ? (
        <motion.div
          key="pill"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={swapTransition}
          className="relative z-10 flex items-center gap-2 py-1.5 pl-1.5 pr-1.5"
        >
          <PhaseGlyph phase={phase} fraction={fraction} color={tone.color} chipClass={tone.chip} />
          {percentLabel && (
            <span className="mono min-w-[3.25rem] text-xs font-semibold">{percentLabel}</span>
          )}
          {statusLine(Boolean(percentLabel))}
          <IconButton label={t("downloads:widget.expand")} onClick={onToggleCollapse}>
            <ChevronUp className="h-4 w-4" />
          </IconButton>
        </motion.div>
      ) : (
        <motion.div
          key="panel"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={swapTransition}
          className="relative z-10 p-3.5"
        >
          <div className="flex items-start gap-3">
            <PhaseGlyph
              phase={phase}
              fraction={fraction}
              color={tone.color}
              chipClass={tone.chip}
            />
            <div className="min-w-0 flex-1 pt-px">
              <p className="truncate text-sm font-semibold leading-snug">{modelName}</p>
              {statusLine(false)}
            </div>
            <div className="flex items-center gap-1.5">
              {active && (
                <IconButton label={t("common:actions.cancel")} onClick={onCancel} danger>
                  <X className="h-4 w-4" />
                </IconButton>
              )}
              <IconButton label={t("downloads:widget.collapse")} onClick={onToggleCollapse}>
                <ChevronDown className="h-4 w-4" />
              </IconButton>
            </div>
          </div>

          {showBar && (
            <div className="mt-3 flex items-center gap-3">
              <div
                role="progressbar"
                aria-label={t("downloads:widget.progressLabel")}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.floor(pct)}
                className={`relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06] ${
                  phase === "queued" || phase === "finalizing" ? "motion-safe:animate-pulse" : ""
                }`}
              >
                <div
                  className={`absolute inset-y-0 left-0 rounded-full bg-gradient-to-r ${tone.fill} transition-[width] duration-300 ease-out`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              {percentLabel && (
                <span className="mono w-12 shrink-0 text-right text-xs font-semibold">
                  {percentLabel}
                </span>
              )}
            </div>
          )}

          {(bytesLabel || speedLabel || eta) && (
            <div className="mono mt-1.5 flex items-center justify-between gap-3 text-[11px] text-[var(--ink-faint)]">
              <span className="flex min-w-0 items-center truncate">
                {bytesLabel && <span>{bytesLabel}</span>}
                {bytesLabel && speedLabel && (
                  <span aria-hidden="true" className="mx-1.5 opacity-60">
                    ·
                  </span>
                )}
                {speedLabel && <span>{speedLabel}</span>}
              </span>
              {eta && (
                <span className="shrink-0">{t("downloads:widget.remaining", { time: eta })}</span>
              )}
            </div>
          )}

          {phase === "completed" && (
            <p className="mt-2 text-xs text-[var(--ink-dim)]">
              {t("downloads:widget.completedHint")}
            </p>
          )}

          {message && (
            <p
              className={`mt-2 text-xs leading-relaxed ${
                phase === "failed" ? "text-red-200/90" : "text-[var(--ink-dim)]"
              }`}
            >
              {message}
            </p>
          )}

          {DISMISSABLE_PHASES.has(phase) && (
            <div className="mt-3 flex justify-end">
              <button
                type="button"
                onClick={onDismiss}
                className={[
                  "rounded-full px-3.5 py-1.5 text-xs font-semibold transition active:scale-95",
                  "bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/60",
                ].join(" ")}
              >
                {t("downloads:widget.dismiss")}
              </button>
            </div>
          )}
        </motion.div>
      )}
    </motion.section>
  );
}

DownloadWidget.propTypes = {
  modelName: PropTypes.string,
  phase: PropTypes.oneOf(DOWNLOAD_PHASES).isRequired,
  /** 0-100, as reported by the status poll. */
  progress: PropTypes.number,
  /** Estimated seconds remaining; 0 / null when the backend has no estimate. */
  timeLeft: PropTypes.number,
  /** `total_bytes` from the status poll; 0 while unmeasured. */
  totalBytes: PropTypes.number,
  /** Derived by the context from progress x total; null while unmeasured. */
  downloadedBytes: PropTypes.number,
  /** Derived by the context from the backend's ETA; null while unmeasured. */
  speedBytesPerSec: PropTypes.number,
  /** Failure / stall / cancel detail, already translated. */
  message: PropTypes.string,
  collapsed: PropTypes.bool,
  onToggleCollapse: PropTypes.func.isRequired,
  onCancel: PropTypes.func.isRequired,
  onDismiss: PropTypes.func.isRequired,
};
