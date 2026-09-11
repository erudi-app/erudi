import React from "react";
import PropTypes from "prop-types";
import { HelpCircle, Cpu, AlertTriangle } from "lucide-react";
import { Trans, useTranslation } from "react-i18next";
import logoErudi from "../../assets/images/logos/logoerudifinal.png";
import { formatPercent } from "../../i18n/format";

WelcomeModal.propTypes = {
  isOpen: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  loading: PropTypes.bool,
  // The startup hardware readout. Absent until the benchmark answers, and the
  // window may be missing from it when profiling fell back.
  hardwareInfo: PropTypes.shape({
    global_inference_score: PropTypes.number,
    global_inference_label: PropTypes.string,
    recommended_param_min: PropTypes.number,
    recommended_param_max: PropTypes.number,
    error: PropTypes.string,
  }),
};

// Badge colors for the labels the backend actually emits
// (hardware/services.py _get_label: Excellent / Good / Fair / Poor / Weak).
const COLOR_BY_LABEL = {
  Excellent: "bg-emerald-700/30 text-white",
  Good: "bg-green-600/30 text-white",
  Fair: "bg-yellow-600/30 text-white",
  Poor: "bg-red-600/30 text-white",
  Weak: "bg-red-600/30 text-white",
};

// The caption below the score derives from the SAME thresholds as the label
// badge (80+ Excellent, 60+ Good, 40+ Fair, below Poor/Weak) so the two
// never contradict each other — a 53% "Fair" must not read "Great
// Performance!" (#303).
const recommendationTier = (inferenceScore) => {
  if (inferenceScore >= 80) return "excellent";
  if (inferenceScore >= 60) return "good";
  if (inferenceScore >= 40) return "fair";
  return "limited";
};

export default function WelcomeModal({ isOpen, onClose, hardwareInfo, loading }) {
  const { t } = useTranslation();

  if (!isOpen) {
    return null;
  }

  // The backend emits the label in English; show it in the app language when
  // it is one of the known tiers, verbatim otherwise.
  const labelText = (label) =>
    label
      ? t(`landing:welcome.label.${String(label).toLowerCase()}`, { defaultValue: label })
      : t("common:status.unknown");

  const tier = hardwareInfo ? recommendationTier(hardwareInfo.global_inference_score) : null;

  // The size window comes from the benchmark, the same numbers the machine
  // readout shows behind this dialog. The tier copy above is qualitative on
  // purpose: baking a range into it let the two disagree on the first screen a
  // user ever sees (#523). Absent when profiling could not produce a window.
  const windowMin = hardwareInfo?.recommended_param_min;
  const windowMax = hardwareInfo?.recommended_param_max;
  const hasWindow =
    typeof windowMin === "number" &&
    typeof windowMax === "number" &&
    Number.isFinite(windowMin) &&
    Number.isFinite(windowMax) &&
    windowMax > 0;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className={[
          "rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto",
          "border border-white/10",
          "bg-[rgba(22,40,36,0.45)] backdrop-blur-[18px] saturate-[1.4]",
          "shadow-[0_8px_30px_-4px_rgba(0,0,0,0.45),0_2px_6px_-1px_rgba(0,0,0,0.4),inset_0_1px_0_rgba(255,255,255,0.06)]",
        ].join(" ")}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="text-center py-6 px-6 sm:py-8 sm:px-8">
          <h1 className="text-4xl sm:text-5xl font-bold mb-4">
            <span className="text-[#00B574]">{t("landing:welcome.title")}</span>
          </h1>
          <p className="text-lg sm:text-xl text-gray-300 mb-2 flex items-center justify-center gap-2">
            <Trans
              i18nKey="landing:welcome.intro"
              components={{
                logo: <img src={logoErudi} alt="erudi" className="h-7 sm:h-7 -mt-2" />,
                accent: <span className="text-[#00B574]" />,
              }}
            />
          </p>
          <p className="text-lg sm:text-xl text-gray-300">
            <Trans
              i18nKey="landing:welcome.tagline"
              components={{ accent: <span className="text-[#00B574]" /> }}
            />
          </p>
        </div>

        {/* Content */}
        <div className="px-4 pb-6 sm:px-8 sm:pb-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
            {/* Left Column - Important Notice */}
            <div className="bg-amber-900/30 border border-amber-600/40 rounded-xl p-4 sm:p-6">
              <div className="flex items-start gap-3 sm:gap-4">
                <AlertTriangle className="w-8 h-8 text-[#E5D07D] mt-1" />
                <div className="flex-1">
                  <h3 className="text-[#E5D07D] font-semibold text-lg mb-3 flex items-center gap-2">
                    {t("landing:welcome.notice.title")}
                  </h3>
                  <div className="space-y-3 text-sm sm:text-base">
                    <p className="text-gray-300 leading-relaxed">
                      {t("landing:welcome.notice.local")}
                    </p>
                    <p className="text-gray-300 leading-relaxed">
                      {t("landing:welcome.notice.earlyRelease")}
                    </p>
                    <p className="text-gray-300 leading-relaxed">
                      {t("landing:welcome.notice.feedback")}
                    </p>
                    <p className="text-[#E5D07D] font-bold">
                      {t("landing:welcome.notice.closing")}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column - Hardware Evaluation */}
            <div className="space-y-4">
              {/* Hardware Evaluation */}
              <div className="bg-[#1A1A1A]/70 border border-white/10 rounded-xl p-4 sm:p-6 backdrop-blur-[10px] saturate-[1.2]">
                <div className="flex items-center gap-3 mb-4">
                  {/* Remove the container div and use a larger CPU icon */}
                  <Cpu className="w-8 h-8 text-[#00B574]" />
                  <h3 className="text-[#00B574] font-semibold text-lg">
                    {t("landing:welcome.hardware.title")}
                  </h3>
                </div>

                {loading ? (
                  <div className="flex items-center justify-center py-6 sm:py-8">
                    <div className="w-6 h-6 sm:w-8 sm:h-8 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
                    <span className="ml-3 text-gray-300 text-sm sm:text-base">
                      {t("landing:welcome.hardware.evaluating")}
                    </span>
                  </div>
                ) : hardwareInfo?.error ? (
                  <div className="text-red-400 bg-red-900/20 border border-red-600/30 rounded-lg p-3">
                    <p className="font-medium">{t("landing:welcome.hardware.failed")}</p>
                    <p className="text-sm mt-1">{hardwareInfo.error}</p>
                  </div>
                ) : hardwareInfo ? (
                  <div className="space-y-3">
                    {/* Performance Cards */}
                    <div className="space-y-3">
                      <div className="bg-[#242424]/60 border border-white/10 rounded-lg p-3 sm:p-4 backdrop-blur-[8px] saturate-[1.1]">
                        <div className="flex items-center justify-between">
                          <span className="text-gray-400 text-sm">
                            {t("landing:welcome.hardware.chatPerformance")}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="text-lg sm:text-xl font-bold text-white">
                              {formatPercent(hardwareInfo.global_inference_score, {
                                maximumFractionDigits: 0,
                              })}
                            </span>
                            <span
                              className={`px-2 py-1 rounded-full text-xs font-medium ${
                                COLOR_BY_LABEL[hardwareInfo.global_inference_label] ||
                                "bg-gray-600/30 text-white"
                              }`}
                            >
                              {labelText(hardwareInfo.global_inference_label)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Recommendations Summary */}
                    {tier && (
                      <div className="bg-[#242424]/60 border border-white/10 rounded-lg p-3 sm:p-4 backdrop-blur-[8px] saturate-[1.1]">
                        <div className="flex items-start gap-3">
                          <HelpCircle className="w-4 h-4 sm:w-5 sm:h-5 text-orange-300 transition-colors cursor-help mt-0.5" />
                          <div className="flex-1 min-w-0">
                            <h4 className="text-orange-300 font-semibold mb-2">
                              {t(`landing:welcome.recommendation.${tier}.title`)}
                            </h4>
                            <p className="text-gray-300 text-sm leading-relaxed">
                              {t(`landing:welcome.recommendation.${tier}.description`)}
                            </p>
                            {hasWindow && (
                              <p className="text-gray-300 text-sm leading-relaxed mt-2">
                                {t("landing:welcome.recommendation.window", {
                                  min: windowMin,
                                  max: windowMax,
                                })}
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          {/* Get Started Button - Centered */}
          <div className="flex justify-center mt-6">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClose();
              }}
              className={[
                "rounded-full px-5 py-2 text-sm font-semibold",
                "bg-[#00B574]/80 hover:bg-[#009960]/80 text-white",
                "border border-white/20 shadow backdrop-blur-[6px] saturate-[1.1]",
                "transition active:scale-95",
                "flex items-center gap-2",
              ].join(" ")}
            >
              {t("landing:welcome.getStarted")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
