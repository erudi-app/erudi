import { useState, useEffect, useCallback } from "react";
import { apiClient } from "../../../services/api/client";
import { createLogger } from "../../../utils/logger";
import { isBackendReady } from "../../../utils/backendStatus";

const log = createLogger("APIHooks");

/**
 * Custom hook for fetching local LLMs
 * @returns {Object} { llms, loading, error }
 */
export function useLLMs() {
  const [llms, setLlms] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchLlms = async () => {
      try {
        setLoading(true);
        const data = await apiClient.get("/llms/local");
        setLlms(Array.isArray(data) ? data : []);
        setError(null);
      } catch (err) {
        log.error("Failed to fetch LLMs", err);
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchLlms();
  }, []);

  return { llms, loading, error };
}

/**
 * Custom hook for fetching a single conversation
 * @param {string} conversationId - Conversation ID
 * @returns {Object} { conversation, loading, error }
 */
export function useConversation(conversationId) {
  const [conversation, setConversation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!conversationId) {
      return;
    }

    const fetchConversation = async () => {
      try {
        setLoading(true);
        const data = await apiClient.get(`/conversations/${conversationId}`);
        setConversation(data);
        setError(null);
      } catch (err) {
        log.error(`Failed to fetch conversation ${conversationId}`, err);
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchConversation();
  }, [conversationId]);

  return { conversation, loading, error };
}

/**
 * Custom hook for fetching conversation messages
 * @param {string} conversationId - Conversation ID
 * @returns {Object} { messages, loading, error }
 */
export function useConversationMessages(conversationId) {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!conversationId) {
      return;
    }

    const fetchMessages = async () => {
      try {
        setLoading(true);
        const data = await apiClient.get(`/conversations/${conversationId}/fetch_messages`);
        setMessages(Array.isArray(data) ? data : []);
        setError(null);
      } catch (err) {
        log.error(`Failed to fetch messages for conversation ${conversationId}`, err);
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchMessages();
  }, [conversationId]);

  return { messages, loading, error };
}

/**
 * Custom hook for fetching all conversations
 * @returns {Object} { conversations, loading, error }
 */
export function useConversations() {
  const [conversations, setConversations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refetch = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get("/conversations/");
      setConversations(Array.isArray(data) ? data : []);
      setError(null);
    } catch (err) {
      log.error("Failed to fetch conversations", err);
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { conversations, loading, error, refetch };
}

/**
 * Custom hook for hardware app startup info.
 *
 * `backend_type` names the engine that RUNS, so the info goes stale when the
 * backend restarts on another engine (the Inference engine setting). The hook
 * fetches again each time the main process reports the backend ready
 * (`backend_ready`, through `window.backendAPI.onBackendEvent`; no bridge
 * outside Electron, so no re-read there) and counts those re-reads once they
 * settle, success or failure, in `refetchesAfterReady`.
 *
 * App.jsx subscribed to the same events first (it mounts before any page), so
 * by the time this listener runs on a `backend_ready` the API client already
 * points at the restarted backend's port.
 *
 * @returns {Object} { startupInfo, loading, error, refetch, refetchesAfterReady }
 */
export function useAppStartupInfo() {
  const [startupInfo, setStartupInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refetchesAfterReady, setRefetchesAfterReady] = useState(0);

  const refetch = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiClient.get("/hardware/app_startup");
      setStartupInfo(data);
      setError(null);
    } catch (err) {
      log.error("Failed to fetch app startup info", err);
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  useEffect(() => {
    const subscribe = window.backendAPI?.onBackendEvent;
    if (typeof subscribe !== "function") return undefined;
    let active = true;
    const unsubscribe = subscribe((evt) => {
      if (!isBackendReady(evt)) return;
      refetch().then(() => {
        if (active) setRefetchesAfterReady((n) => n + 1);
      });
    });
    return () => {
      active = false;
      if (typeof unsubscribe === "function") unsubscribe();
    };
  }, [refetch]);

  return { startupInfo, loading, error, refetch, refetchesAfterReady };
}

/**
 * Custom hook for backend health check
 * @returns {Object} { isHealthy, loading, error }
 */
export function useBackendHealth() {
  const [isHealthy, setIsHealthy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        setLoading(true);
        await apiClient.get("/health/");
        setIsHealthy(true);
        setError(null);
      } catch (err) {
        log.error("Backend health check failed", err);
        setIsHealthy(false);
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    checkHealth();
  }, []);

  return { isHealthy, loading, error };
}

/**
 * Custom hook for the user-settings singleton (GET/PUT /user_settings/).
 * Carries the global web-search default; new conversations inherit it.
 * @returns {Object} { settings, loading, error, updateSettings }
 */
export function useUserSettings() {
  const [settings, setSettings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        setLoading(true);
        const data = await apiClient.get("/user_settings/");
        setSettings(data);
        setError(null);
      } catch (err) {
        log.error("Failed to fetch user settings", err);
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchSettings();
  }, []);

  const updateSettings = useCallback(async (payload) => {
    try {
      const data = await apiClient.put("/user_settings/", payload);
      setSettings(data);
      setError(null);
      return data;
    } catch (err) {
      log.error("Failed to update user settings", err);
      setError(err);
      throw err;
    }
  }, []);

  return { settings, loading, error, updateSettings };
}
