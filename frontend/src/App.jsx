import React, { useState, useEffect, useCallback } from "react";
import { HashRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import ChatPage from "./pages/ChatPage";
import ConversationPage from "./pages/ConversationPage";
import ArenaPage from "./pages/ArenaPage";
import KnowledgeBasePage from "./pages/KnowledgeBasePage";
import SettingsPage from "./pages/SettingsPage";
import { DownloadModalProvider } from "./contexts/DownloadModalContext";
import { KnowledgeBaseProvider } from "./contexts/KnowledgeBaseContext";
import LoadingScreen from "./components/LoadingScreen";
import BackendErrorScreen from "./components/BackendErrorScreen";
import UpdateBanner from "./components/UpdateBanner";
import EngineFailureModal from "./components/modals/EngineFailureModal";
import InteractionLogger from "./components/InteractionLogger";
import { apiClient } from "./services/api/client";
import { setBackendPort } from "./config/api";
import { syncLanguageWithBackend } from "./i18n/sync";
import { syncAutoUpdateWithMain } from "./utils/autoUpdate";
import {
  describeBackendError,
  isStartupError,
  isBackendReady as isReadyEvent,
} from "./utils/backendStatus";
import { isEngineNotice } from "./utils/engineNotice";
import { createLogger } from "./utils/logger";

const log = createLogger("App");

// Fallback only (no Electron preload, e.g. a browser/e2e context): poll /health
// and give up after this long. In the real app, readiness is event-driven —
// main.js waits patiently for the backend and never kills it for being slow.
const FALLBACK_HEALTH_TIMEOUT_MS = 90000;

export default function App() {
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [backendError, setBackendError] = useState(null);
  const [phase, setPhase] = useState(null);
  const [firstRun, setFirstRun] = useState(false);
  const [retryNonce, setRetryNonce] = useState(0);
  // The backend's CUDA pre-flight found a graphics card it cannot drive. This
  // is NOT a startup failure — the app is fine, the first message would not be
  // — so it is held here and shown over the real UI once the app is past
  // `ready`, never on the loader and never on the error screen.
  const [engineNotice, setEngineNotice] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const bridge = window.backendAPI;

    // Real app: trust the main process, which owns the backend, knows the
    // resolved port, waits for `ready`, and confirms health. We react to its
    // forwarded events and also query getInfo() to cover the race where
    // readiness happened before this listener attached.
    if (bridge?.onBackendEvent) {
      const unsubscribe = bridge.onBackendEvent((evt) => {
        if (cancelled) return;
        if (isStartupError(evt)) {
          // The main process wrote the ERROR record for this event (it owns
          // the backend's lifecycle); this side only notes the screen change.
          log.info("Backend startup error received; showing the error screen", evt);
          setBackendError(describeBackendError(evt));
          return;
        }
        if (isEngineNotice(evt)) {
          log.warn("Engine notice from the backend", evt);
          setEngineNotice(evt);
          return;
        }
        if (evt?.port) setBackendPort(evt.port);
        if (evt?.event === "starting") setFirstRun(!!evt.first_run);
        if (evt?.event === "phase") setPhase(evt.phase);
        if (isReadyEvent(evt)) {
          log.log("Backend is ready");
          setIsBackendReady(true);
        }
      });

      bridge
        .getInfo?.()
        .then((info) => {
          if (cancelled || !info) return;
          if (info.port) setBackendPort(info.port);
          if (info.ready) setIsBackendReady(true);
        })
        .catch((error) => {
          // The events still arrive; only the catch-up read is lost.
          log.warn("Could not read the backend state from the main process", error);
        });

      return () => {
        cancelled = true;
        if (typeof unsubscribe === "function") unsubscribe();
      };
    }

    // Fallback (no preload bridge): poll /health directly.
    let timer = null;
    const start = Date.now();
    const timeoutMs = Number(window.__ERUDI_BACKEND_TIMEOUT_MS__) || FALLBACK_HEALTH_TIMEOUT_MS;
    const poll = async () => {
      if (cancelled) return;
      try {
        await apiClient.get("/health/");
        if (!cancelled) setIsBackendReady(true);
      } catch (error) {
        if (cancelled) return;
        if (Date.now() - start >= timeoutMs) {
          setBackendError((prev) => prev || describeBackendError({ code: "BACKEND_UNREACHABLE" }));
          return;
        }
        timer = setTimeout(poll, 2000);
      }
    };
    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [retryNonce]);

  // Once the backend answers, reconcile the boot language (local mirror or
  // OS locale) with the persisted user setting (#385), and hand the persisted
  // automatic-update preference to the main process, which holds every update
  // check until it hears one.
  useEffect(() => {
    if (!isBackendReady) return;
    syncLanguageWithBackend(apiClient);
    syncAutoUpdateWithMain(apiClient);
  }, [isBackendReady]);

  const handleRetry = useCallback(() => {
    setBackendError(null);
    setIsBackendReady(false);
    setPhase(null);
    // Actually re-spawn the backend (not just re-poll) when the bridge exists.
    window.backendAPI?.restartBackend?.().catch((error) => {
      log.error("The backend restart the user asked for did not happen", error);
    });
    setRetryNonce((n) => n + 1);
  }, []);

  const handleQuit = useCallback(() => {
    window.close();
  }, []);

  if (backendError) {
    return <BackendErrorScreen error={backendError} onRetry={handleRetry} onQuit={handleQuit} />;
  }

  if (!isBackendReady) {
    return <LoadingScreen phase={phase} firstRun={firstRun} />;
  }

  return (
    <DownloadModalProvider>
      <KnowledgeBaseProvider>
        <UpdateBanner />
        {/* Dismissing persists nothing: the condition is real and still there
            next launch, so the notice is expected to come back. */}
        <EngineFailureModal notice={engineNotice} onDismiss={() => setEngineNotice(null)} />
        <Router>
          {/* Mounted-once UI interaction tracer (needs the Router for useLocation). */}
          <InteractionLogger />
          <Routes>
            <Route path="/" element={<Navigate to="/erudi/models" replace />} />
            <Route path="/erudi" element={<Navigate to="/erudi/models" replace />} />
            <Route path="*" element={<Navigate to="/erudi/models" replace />} />
            <Route path="/erudi/chat" element={<ChatPage />} />
            <Route path="/erudi/models" element={<LandingPage />} />
            <Route path="/erudi/conversations/:id" element={<ConversationPage />} />
            <Route path="/erudi/arena" element={<ArenaPage />} />
            <Route path="/erudi/attach_knowledge_base" element={<KnowledgeBasePage />} />
            <Route path="/erudi/settings" element={<SettingsPage />} />
          </Routes>
        </Router>
      </KnowledgeBaseProvider>
    </DownloadModalProvider>
  );
}
