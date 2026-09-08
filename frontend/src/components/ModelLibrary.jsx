import React from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";
import { RefreshCcw } from "lucide-react";

/**
 * Model Library component for selecting and managing local models
 *
 * @param {Object} props
 * @param {Array} props.models - Array of available models
 * @param {string|null} props.selectedModel - Currently selected model ID
 * @param {function} props.onModelSelect - Callback when a model is selected
 * @param {string} props.modelName - Name for the new model
 * @param {function} props.onModelNameChange - Callback when model name changes
 * @param {function} props.onRefresh - Callback to refresh the models list
 */

ModelLibrary.propTypes = {
  models: PropTypes.arrayOf(PropTypes.object),
  isLoading: PropTypes.bool,
  onModelClick: PropTypes.func,
  onModelDownload: PropTypes.func,
};

ModelLibrary.defaultProps = {
  models: [],
  isLoading: false,
  onModelClick: null,
  onModelDownload: null,
};

export default function ModelLibrary({
  models = [],
  selectedModel,
  onModelSelect,
  modelName = "",
  onModelNameChange,
  onRefresh,
}) {
  const { t } = useTranslation();

  return (
    <div className="flex-1 min-w-[300px] bg-[#2B2B2B] rounded-2xl p-6 text-white shadow-lg flex flex-col gap-4 border border-white/20 border-[0.5px]">
      <div className="flex items-center justify-between">
        <h3 className="text-xl md:text-2xl font-bold">{t("models:library.title")}</h3>
        <RefreshCcw
          className="w-4 h-4 cursor-pointer hover:rotate-90 transition"
          onClick={onRefresh}
          title={t("models:library.refresh")}
        />
      </div>
      <div
        className="bg-[#242323] rounded-lg p-3 overflow-y-auto max-h-40 shadow-lg border border-white/20 border-[0.5px]"
        style={{
          scrollbarWidth: "thin",
          scrollbarColor: "#9CA3AF #374151",
        }}
      >
        {models.length === 0 ? (
          <div className="text-white/60 text-sm">{t("models:library.empty")}</div>
        ) : (
          <div className="space-y-1.5">
            {models.map((model) => (
              <div
                key={model.id}
                onClick={() => onModelSelect(model.id)}
                className={`
                  relative px-3 py-2 rounded-lg border transition-all duration-200 cursor-pointer
                  ${
                    selectedModel === model.id
                      ? "bg-emerald-400/10 border-emerald-400/30 text-emerald-300"
                      : "bg-[#3A3A3A] border-gray-600/50 text-gray-300 hover:bg-[#404040] hover:border-gray-500/70"
                  }
                `}
              >
                {/* Selection indicator */}
                {selectedModel === model.id && (
                  <div className="absolute left-1 top-1/2 transform -translate-y-1/2 w-1 h-6 bg-emerald-400 rounded-full" />
                )}

                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0 ml-2">
                    <h5 className="font-medium text-sm truncate">{model.name}</h5>
                    {/* Show model type if available */}
                    {model.type && (
                      <span className="text-xs text-gray-400 mt-0.5 block">{model.type}</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Model Name Input - moved below the list */}
      <div className="border-t border-gray-600/30 pt-4">
        <div className="mb-2">
          <h4 className="text-sm font-semibold text-white/90">
            {t("models:library.newNameTitle")}
          </h4>
          <p className="text-xs text-white/60">{t("models:library.newNameHint")}</p>
        </div>
        <div className="flex gap-2">
          <input
            className="flex-1 border rounded-lg px-3 py-2 text-sm placeholder-white/40 focus:ring-0 focus:outline-none transition-colors bg-[#3A3A3A] border-gray-600/50 text-white focus:border-emerald-400/50 focus:bg-[#404040]"
            placeholder={
              selectedModel ? t("models:library.namePlaceholder") : t("models:library.selectFirst")
            }
            value={modelName}
            onChange={(e) => onModelNameChange(e.target.value)}
            disabled={!selectedModel}
          />
        </div>
      </div>
    </div>
  );
}
