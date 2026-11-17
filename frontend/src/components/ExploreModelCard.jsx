import React from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import { AlertTriangle, BadgeCheck, Download, Heart, Image as ImageIcon } from "lucide-react";
import GradientBox from "./GradientBox";
import FitGauge from "./FitGauge";
import { CATEGORY_META } from "../utils/modelCatalog";
import {
  modelSupportsVision,
  isVerySmallModel,
  SMALL_MODEL_PARAM_THRESHOLD_B,
} from "../utils/modelCapabilities";
import { isTestedModel } from "../utils/testedModels";
import { publishedLabel } from "../utils/publishedDate";

/**
 * Explore-panel model card. Frosted-glass surface (the look the catalog has always
 * had), no decorative icons — the fit gauge is the only graphic, because "will it
 * run on my machine?" is the one thing a local-LLM user needs at a glance. Name and
 * category up top, the gauge in the middle, monospace metrics, one clear action.
 */
export default function ExploreModelCard({ model, range, onDownload, onInfo, installed = false }) {
  const { t } = useTranslation();
  const unavailable = model?.runnable === false;
  const isVision = modelSupportsVision(model);
  const verySmall = isVerySmallModel(model);
  const tested = isTestedModel(model);
  const cat = CATEGORY_META[model.category];
  // Compact parameter count ("7B", "270M"): a technical label, not copy.
  const params =
    typeof model.param_size === "number"
      ? model.param_size >= 1
        ? `${Number(model.param_size.toFixed(model.param_size < 10 ? 1 : 0))}B`
        : `${Math.round(model.param_size * 1000)}M`
      : null;

  // Compact download/like counts ("123k", "2M"), the Hugging Face convention.
  const formatCount = (val) => {
    const n = parseInt(String(val ?? "").replace(/[^\d]/g, ""), 10);
    if (!n) return null;
    if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1).replace(/\.0$/, "")}M`;
    if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
    return String(n);
  };

  const downloads = formatCount(model.downloads);
  const likes = formatCount(model.likes);
  const published = publishedLabel(model.publishedAt);

  return (
    <GradientBox
      className="h-full bg-[#1a1a1a]/60 backdrop-blur-sm border border-white/10 transition-colors duration-200 hover:border-[var(--fit-good)]/40 after:pointer-events-none after:absolute after:inset-0 after:z-[5] after:bg-[var(--fit-good)] after:opacity-0 after:transition-opacity after:duration-200 hover:after:opacity-[0.07]"
      // eslint-disable-next-line i18next/no-literal-string -- Tailwind class list, not copy
      contentClassName="relative z-10 flex flex-col h-full p-4"
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-1.5 min-w-0">
          <h4 className="text-[15px] font-semibold text-[var(--ink)] leading-snug">{model.name}</h4>
          {tested && (
            <span
              className="flex items-center shrink-0 text-[var(--fit-good)]"
              title={t("models:card.testedBadge")}
            >
              <BadgeCheck className="w-4 h-4" />
            </span>
          )}
        </div>
        {cat && (
          <span className="eyebrow !text-[9px] !tracking-[0.12em] whitespace-nowrap pt-1 text-[var(--ink-faint)]">
            {t(cat.labelKey)}
          </span>
        )}
      </div>

      <FitGauge
        paramSize={model.param_size}
        quantized={model.quantized}
        sizeBytes={model.artifact_size_bytes}
        range={range}
        showLabel={!unavailable}
      />

      {unavailable && (
        <div className="mt-1.5 mono text-[11px] text-[var(--fit-heavy)]">
          {t("models:explore.notSupported")}
        </div>
      )}

      {/* Very small models (#381): say on the card, before the download, that
          tools, KB search and multi-step reasoning will not work reliably, so
          a silent no-tool-call later does not read as a broken feature. */}
      {verySmall && (
        <p
          data-testid="small-model-note"
          className="mt-2 flex items-start gap-1.5 text-[11px] leading-snug text-[var(--ink-dim)]"
        >
          <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-px text-[var(--fit-tight)]" />
          <span>{t("models:smallModelNote", { threshold: SMALL_MODEL_PARAM_THRESHOLD_B })}</span>
        </p>
      )}

      <div className="mt-auto pt-4">
        <div className="mono text-xs text-[var(--ink-dim)] flex items-center gap-2 mb-2.5">
          {/* Unmeasured size (#201): say so plainly rather than implying a value. */}
          {params ? (
            <span>{params}</span>
          ) : (
            <span className="italic">{t("models:explore.sizeUnknown")}</span>
          )}
          {isVision && (
            <span
              className="flex items-center gap-1 text-[var(--fit-good)]"
              title={t("models:explore.visionBadge")}
            >
              <ImageIcon className="w-3.5 h-3.5" />
            </span>
          )}
          {downloads && (
            <span className="flex items-center gap-1">
              <Download className="w-3 h-3" />
              <span className="text-[var(--ink)] font-semibold">{downloads}</span>
            </span>
          )}
          {likes && (
            <span className="flex items-center gap-1">
              <Heart className="w-3 h-3" />
              <span className="text-[var(--ink)] font-semibold">{likes}</span>
            </span>
          )}
          {model.gated && (
            <span className="text-[var(--fit-tight)]">{t("models:explore.gated")}</span>
          )}
        </div>
        {/* Nothing at all when the publication date is unknown (#561). */}
        {published && <div className="mono text-xs text-[var(--ink-dim)] mb-2.5">{published}</div>}
        <div className="flex items-center justify-end gap-3">
          <button
            onClick={() => onInfo && onInfo(model)}
            className="text-sm text-[var(--ink-dim)] hover:text-[var(--ink)] transition-colors"
          >
            {t("models:explore.details")}
          </button>
          {/* An installed model must not keep offering its own download (#348):
              it reads as "I downloaded it and it is still asking me to", and on
              a machine short of disk the natural worry is whether clicking it
              fetches several gigabytes a second time. */}
          <button
            onClick={() => !unavailable && !installed && onDownload && onDownload(model)}
            disabled={unavailable || installed}
            title={installed ? t("models:explore.alreadyInstalled") : undefined}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
              unavailable || installed
                ? "opacity-30 cursor-not-allowed text-[var(--ink-dim)]"
                : "bg-[var(--fit-good)] text-[#07241d] hover:brightness-110"
            }`}
          >
            {installed ? t("models:explore.installed") : t("common:actions.download")}
          </button>
        </div>
      </div>
    </GradientBox>
  );
}

ExploreModelCard.propTypes = {
  model: PropTypes.object.isRequired,
  range: PropTypes.shape({ min: PropTypes.number, max: PropTypes.number }),
  onDownload: PropTypes.func,
  onInfo: PropTypes.func,
  installed: PropTypes.bool,
};
