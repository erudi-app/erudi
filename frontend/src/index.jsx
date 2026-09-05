import * as React from "react";
import { createRoot } from "react-dom/client";
// Initializes i18next synchronously (bundled catalogs, language from the
// local mirror or the OS locale) before the first render (#385).
import "./i18n";
import App from "./App.jsx";
// Uncaught errors and unhandled rejections reach no log without this, and the
// window goes blank with nothing to report. Installed before the first render
// so an exception thrown during boot is caught too.
import { installGlobalErrorCapture } from "./utils/errorCapture";
// Wraps the whole tree, boot screens included: a render that throws leaves the
// window blank otherwise, and React does not route it to window.onerror.
import AppErrorBoundary from "./components/AppErrorBoundary";
// Self-hosted Montserrat (latin), bundled so the app's typography works offline.
import "@fontsource/montserrat/latin-400.css";
import "@fontsource/montserrat/latin-500.css";
import "@fontsource/montserrat/latin-600.css";
import "@fontsource/montserrat/latin-700.css";
import "@fontsource/montserrat/latin-800.css";
import "./index.css";

installGlobalErrorCapture();

function renderApp() {
  const container = document.getElementById("root");
  if (!container) {
    return;
  }
  const root = createRoot(container);
  root.render(
    <AppErrorBoundary>
      <App />
    </AppErrorBoundary>
  );

  const loader = document.getElementById("loader");
  if (loader) {
    loader.style.transition = "opacity 0.5s ease";
    loader.style.opacity = "0";
    setTimeout(() => loader.remove(), 10);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", renderApp);
} else {
  renderApp();
}
