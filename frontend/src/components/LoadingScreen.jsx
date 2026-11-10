import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";

// The backend's startup-progress phases (backend/run.py + the FastAPI
// lifespan emit these). Each maps to a `common:boot.phase.*` label; unknown
// or absent phases show the generic "Starting" one.
const KNOWN_PHASES = [
  "starting",
  // Sent by the Electron main process, not the backend: the previous
  // backend still holds the data folder (DATA_DIR_IN_USE) and main waits.
  "waiting_previous_session",
  "preparing_database",
  "recovering_database",
  "running_migrations",
  "loading_catalog",
  "ready",
];

export default function LoadingScreen({ phase, firstRun }) {
  const { t } = useTranslation();
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);

  const phaseKey = KNOWN_PHASES.includes(phase) ? phase : "starting";
  const label = t(`common:boot.phase.${phaseKey}`);

  return (
    <div
      className="fixed top-0 left-0 w-screen h-screen flex flex-col justify-center items-center z-[9999] px-8 text-center"
      style={{ backgroundColor: "#02130e" }}
    >
      <img
        src={require("../assets/images/logos/logoerudifinal.png")}
        alt={t("common:boot.logoAlt")}
        className="mb-2 object-contain"
        style={{ maxWidth: "14rem", maxHeight: "14rem" }}
      />
      <p className="text-xl mt-1 mb-8" style={{ color: "#e0e0e0" }}>
        {t("common:tagline")}
      </p>
      <div className="w-12 h-12 border-4 border-gray-200/20 border-t-gray-200/80 rounded-full animate-spin"></div>
      <p className="text-sm mt-6" style={{ color: "#cfd8d4" }}>
        {label}{" "}
        {elapsed > 0 && (
          <span style={{ color: "#7c8f88" }}>{t("common:boot.elapsed", { seconds: elapsed })}</span>
        )}
      </p>
      {firstRun && (
        <p className="text-xs mt-2 max-w-sm" style={{ color: "#9fb0aa" }}>
          {t("common:boot.firstRun")}
        </p>
      )}
    </div>
  );
}

LoadingScreen.propTypes = {
  phase: PropTypes.string,
  firstRun: PropTypes.bool,
};
