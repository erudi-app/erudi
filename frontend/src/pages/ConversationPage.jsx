import React, {
  useEffect,
  useLayoutEffect,
  useState,
  useCallback,
  useRef,
  useSyncExternalStore,
} from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import Sidebar from "../components/Sidebar";
import ChatCollapsibleSection from "../components/ChatCollapsibleSection";
import QuestionInput from "../components/QuestionInput";
import ConversationHeader from "../components/ConversationHeader";
import UnloadModelButton from "../components/UnloadModelButton";
import CommandPopups from "../components/CommandPopups";
import CommandNotice from "../components/CommandNotice";
import ContextUsagePanel from "../components/ContextUsagePanel";
import EngineFailureModal from "../components/modals/EngineFailureModal";
import { AlertTriangle, Copy, Check, FileText, Star } from "lucide-react";
import TypingIndicator from "../components/TypingIndicator";
import MarkdownRenderer from "../components/MarkdownRenderer";
import TraceStrip from "../components/TraceStrip";
import CompactionMarker from "../components/CompactionMarker";
import ErrorDetails from "../components/ErrorDetails";
import { API_BASE_URL } from "../config/api.js";
import apiClient, { tracedFetch } from "../services/api/client";
import { createLogger } from "../utils/logger";
import { conversationPath } from "../utils/routes";
import { canAttachImages, maxImagesForModel } from "../utils/modelCapabilities";
import { COMMAND_SCOPES, commandsFor } from "../utils/slashCommands";
import useConversationSettings from "../shared/hooks/useConversationSettings";
import {
  compactingVersion,
  finishCompacting,
  isCompacting,
  startCompacting,
  subscribeCompacting,
} from "../shared/compactingConversations";
import {
  baseName,
  getAttachmentNames,
  getDisplayContent,
  getImagePaths,
  isGenericFailure,
} from "../utils/messageContent";
import { formatBytes } from "../utils/formatBytes";
import {
  automaticCompactions,
  errorDetailOf,
  errorTurnTrace,
  hasStripEvents,
  manualCompactions,
} from "../utils/traceEvents";

const log = createLogger("ConversationPage");

// Wire sentinel the backend persists on error turns; getDisplayContent and the
// bubble styling detect it, so it stays untranslated and only the text after it
// is localized.
const ERROR_SENTINEL = "[ERROR_MESSAGE_SYSTEM]";

// The red line of a failed turn goes below whatever the model had already
// written, as the backend stores it (a paragraph of its own).
const joinFailure = (partial, failure) => {
  // Exactly one blank line: a notice already ends with one.
  const head = partial.replace(/\n+$/, "");
  return head ? `${head}\n\n${failure}` : failure;
};

// Presentation-only tokens kept out of the JSX tree (they are class names and
// CSS, not copy).
const HIDE_SCROLLBAR_CSS = "::-webkit-scrollbar { display: none; }";
// The amber memory warning's copy, by the `reason` the backend sends.
const MEMORY_WARNING_KEYS = {
  bound: "chat:memoryWarning.bound",
  pressure: "chat:memoryWarning.pressure",
  pressure_external: "chat:memoryWarning.pressureExternal",
  gpu_contention: "chat:memoryWarning.gpuContention",
};
// The reload state's copy, by the `reason` of the `memory_reload` event.
const RELOAD_COPY = {
  memory: "chat:message.memoryReload",
  idle: "chat:message.idleReload",
  manual: "chat:message.manualReload",
};
// While the `gpu_contention` notice shows, the cached contention state is
// polled at this pace (the backend samples it every 10 s too).
const GPU_CONTENTION_STATUS_PATH = "/hardware/gpu_contention/status";
const GPU_CONTENTION_POLL_MS = 10_000;
// The slash commands a conversation offers, in the composer and the header.
const COMMANDS = commandsFor(COMMAND_SCOPES.CONVERSATION);
// While an answer streams, switching model waits for it to end, and so does
// /context (a snapshot taken mid-turn would outlive the turn as if current).
const COMMANDS_DISABLED_WHILE_ANSWERING = ["model", "context"];
// Every command waits while /compact runs.
const ALL_COMMAND_NAMES = COMMANDS.map((command) => command.name);
// A /compact the backend has not answered after this long is given up (the
// summary is bounded server-side; a model load that hangs is not).
const COMPACT_CEILING_MS = 10 * 60 * 1000;
// The re-read that follows it is waited for at most this long, so the
// conversation is always unlocked (a re-read that lands later still shows).
const COMPACT_REREAD_CEILING_MS = 60 * 1000;

/** `promise`, waited for at most `ms`: settles with it, or resolves after `ms`. */
function withinCeiling(promise, ms) {
  let timer;
  const ceiling = new Promise((resolve) => {
    timer = setTimeout(resolve, ms);
  });
  return Promise.race([promise, ceiling]).finally(() => clearTimeout(timer));
}
// What follows a /compact answer (#585): nothing more, the conversation's
// messages read again (its separator, or where it stands after a failure),
// the whole conversation loaded again (its model is gone: the header turns
// to "Please select a model"), or the page left (the conversation is gone).
const AFTER_COMPACT = Object.freeze({
  NONE: "none",
  REREAD: "reread",
  RELOAD: "reload",
  LEAVE: "leave",
});
// The refusals and failures with their own copy, by the backend's
// `detail.reason` or error type, and what follows each.
const COMPACT_ERRORS = {
  no_model: { key: "chat:commands.pickModelFirst", after: AFTER_COMPACT.RELOAD },
  unanswered_question: {
    key: "chat:compact.errors.unansweredQuestion",
    after: AFTER_COMPACT.REREAD,
  },
  ENGINE_BUSY: { key: "chat:compact.errors.busy", after: AFTER_COMPACT.REREAD },
  CONVERSATION_NOT_FOUND: { key: "chat:compact.errors.notFound", after: AFTER_COMPACT.LEAVE },
  COMPACTION_FAILED: { key: "chat:compact.errors.failed", after: AFTER_COMPACT.REREAD },
  COMPACTION_SEPARATOR_NOT_SAVED: {
    key: "chat:compact.errors.separatorNotSaved",
    after: AFTER_COMPACT.REREAD,
  },
};

/**
 * What a /compact answer (`ok`, its JSON `body` or null) says and what
 * follows it: `{ notice: { tone, key, values } | null, after }`. An error
 * without its own copy says the server's message when it sent one ("The
 * summary failed: ..."), else that the summary did not come back.
 */
function compactOutcome(ok, body) {
  if (ok && body?.summarized === 0) {
    return {
      notice: { tone: "info", key: "chat:compact.nothingToCompact" },
      after: AFTER_COMPACT.NONE,
    };
  }
  if (ok && body?.summarized > 0) {
    return { notice: null, after: AFTER_COMPACT.REREAD };
  }
  const error = body?.error;
  const known = COMPACT_ERRORS[error?.detail?.reason] ?? COMPACT_ERRORS[error?.type];
  if (known) {
    return { notice: { tone: "error", key: known.key }, after: known.after };
  }
  if (!ok && typeof error?.message === "string" && error.message.trim()) {
    return {
      notice: {
        tone: "error",
        key: "chat:compact.errors.failedWithMessage",
        values: { message: error.message.trim() },
      },
      after: AFTER_COMPACT.REREAD,
    };
  }
  return {
    notice: { tone: "error", key: "chat:compact.errors.unknown" },
    after: AFTER_COMPACT.REREAD,
  };
}
const alignmentClassFor = (isUser) => (isUser ? "items-end" : "items-start");
const bubbleClassFor = (isUser, isError) => {
  if (isUser) {
    return "bg-[#191919] ml-auto rounded-tr-none text-white";
  }
  return isError ? "text-red-400 mr-auto rounded-tl-none" : " text-white mr-auto rounded-tl-none";
};

// Reconcile a fetched message list with the current one (#303). Fetched DB rows
// win, but trailing LOCAL messages the fetch does not cover yet are kept: the
// in-flight streaming bubble, or a just-finished turn whose DB insert the
// refetch raced. A blind replacement here erased streamed answers (the later
// flushes match by message id and silently no-op once the bubble is gone).
// Kept messages are matched away by content on a later refetch.
const mergeFetchedMessages = (prev, fetched) => {
  const covered = (m) =>
    fetched.some(
      (f) => f.sender === m.sender && (f.content || "").trim() === (m.content || "").trim()
    );
  const tail = [];
  for (let i = prev.length - 1; i >= 0; i--) {
    const m = prev[i];
    if (m.streaming || (m.local && !covered(m))) {
      tail.unshift(m);
    } else {
      break;
    }
  }
  return [...fetched, ...tail];
};

export default function ConversationPage() {
  const { t } = useTranslation();
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const [messages, setMessages] = useState([]);
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [starredIds, setStarredIds] = useState({});
  const [conversations, setConversations] = useState([]);
  const scrollRef = useRef(null);
  // Which conversation the current `messages` state belongs to. Session-only
  // image bytes are preserved from `prev` on a refetch, but ONLY when prev is
  // the same conversation — otherwise a previously-viewed chat's images bleed
  // onto a different chat by matching index/content.
  const loadedConvIdRef = useRef(null);

  // A turn that died because the graphics card could not run the model carries
  // an engine code on its error event. The red bubble stays as it is; the code
  // additionally opens the decision dialog, which is the only place the user
  // can act on it.
  const [engineFailure, setEngineFailure] = useState(null);
  const [initialHandled, setInitialHandled] = useState(false);
  const [loading, setLoading] = useState(false);
  // Turns completed on this page; bumped once each turn is over.
  const [turnsEnded, setTurnsEnded] = useState(0);
  const [models, setModels] = useState([]);
  // The models list and the conversation load independently; the orphan
  // check below only fires once each has resolved, so the "missing model"
  // state never flashes during the initial load.
  const [modelsLoaded, setModelsLoaded] = useState(false);
  // The conversation's model and settings (#609): one owner for the header,
  // the command popups and the send path. Hydrated from the conversation row,
  // every change saved as it is made.
  const conversationSettings = useConversationSettings(id);
  const { adopt, loadStamp, flush, getValues, changeModel } = conversationSettings;
  // The command being run (its popup is open), or null.
  const [openCommand, setOpenCommand] = useState(null);
  // The conversation could not be read: an error with a Retry, which runs
  // the load again (`loadAttempt`); the composer stays locked meanwhile.
  const [loadFailed, setLoadFailed] = useState(false);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [collapsed, setCollapsed] = useState(false);
  const [firstReplyPending, setFirstReplyPending] = useState(false);
  // The amber memory warning (1.1.2): the LATEST turn's `memory_warning`
  // stream event, or null. Transient by design — never persisted, never in
  // the message content — and cleared by the next turn that carries none.
  const [memoryWarning, setMemoryWarning] = useState(null);
  // /compact (#585): whether one runs for this conversation (the session's
  // store, so leaving the page and coming back still shows it locked), the
  // runs this page started, and the one-line outcome of a command above the
  // composer.
  useSyncExternalStore(subscribeCompacting, compactingVersion);
  const compactingHere = isCompacting(id);
  const startedHereRef = useRef(new Set());
  const [commandNotice, setCommandNotice] = useState(null);
  // /context (#608): the usage snapshots drawn at the end of the thread. A
  // list of their own, never in `messages` (whose image carry-over goes by
  // index) and never stored: gone at the next send, re-read after /compact,
  // conversation switch or reload. The epoch moves at each of those, and a
  // usage read under an older epoch is dropped.
  const [contextPanels, setContextPanels] = useState([]);
  const contextPanelSeqRef = useRef(0);
  const contextEpochRef = useRef(0);
  const clearContextPanels = useCallback(() => {
    contextEpochRef.current += 1;
    setContextPanels([]);
  }, []);
  // The conversation on screen now, for an answer that lands later; null once
  // the page is gone.
  const idRef = useRef(id);
  // In the layout phase: after the commit, before any effect of it reads it.
  useLayoutEffect(() => {
    idRef.current = id;
  }, [id]);
  useEffect(() => {
    idRef.current = id;
    return () => {
      idRef.current = null;
    };
  }, [id]);

  // A memory warning is about ONE conversation on THIS machine right now:
  // switching conversations must never show conversation A's warning over B.
  // The state starts over while rendering, so no render shows the previous
  // conversation's warning or notice under the new id; the usage epoch moves
  // in the effect.
  const [seenConversationId, setSeenConversationId] = useState(id);
  if (seenConversationId !== id) {
    setSeenConversationId(id);
    setMemoryWarning(null);
    setOpenCommand(null);
    setLoadFailed(false);
    setCommandNotice(null);
    setContextPanels([]);
  }
  useEffect(() => {
    contextEpochRef.current += 1;
  }, [id]);
  // A ref, not state: the scroll handler updates it on every scroll event, and
  // reading it from the auto-scroll effect must NOT re-run that effect. As state
  // it did (the effect depended on it), so toggling it near the bottom snapped
  // the view back down and the user could not scroll freely.
  const userScrolledUpRef = useRef(false);
  // The pending 100 ms re-check of the scroll position. Held in a ref so that
  // sending a message can cancel it: a re-check that fires after the send
  // would mark the view as scrolled up again.
  const scrollCheckTimerRef = useRef(null);
  // The pending reset of the "copied" tick. Held in a ref so a second copy
  // restarts it and leaving the page cancels it.
  const copiedTimerRef = useRef(null);
  useEffect(() => () => clearTimeout(copiedTimerRef.current), []);

  // Refs to avoid putting messages.length in useCallback deps, which
  // would cascade into the loadConversationData useEffect and overwrite
  // in-flight streaming state with stale DB data.
  const messagesLengthRef = useRef(0);
  useLayoutEffect(() => {
    messagesLengthRef.current = messages.length;
  }, [messages.length]);
  const isStreamingRef = useRef(false);

  // The `gpu_contention` notice says another app is taking the graphics card
  // NOW: while it shows, the cached contention state is polled about every
  // 10 s (a setTimeout chain), and the notice drops once the contention
  // ended (`active === false`; `null`, no reading, is not an end). The chain
  // pauses while a turn runs -- its end re-evaluates the notice -- and a
  // failed poll keeps the notice. The clear is functional
  // and touches a gpu_contention notice only, never a warning of another
  // reason that replaced it meanwhile.
  const contentionShown = memoryWarning?.reason === "gpu_contention";
  useEffect(() => {
    if (!contentionShown) return undefined;
    let cancelled = false;
    let warned = false;
    let timer = null;
    const poll = async () => {
      if (!isStreamingRef.current) {
        try {
          const status = await apiClient.get(GPU_CONTENTION_STATUS_PATH, { silentFailure: true });
          if (cancelled) return;
          if (status?.active === false) {
            setMemoryWarning((current) => (current?.reason === "gpu_contention" ? null : current));
            return;
          }
        } catch (error) {
          // The notice stays: a contention that cannot be read is not over.
          // Said once per notice, not at every tick.
          if (!warned) {
            warned = true;
            log.warn("GPU contention status poll failed", error);
          }
        }
      }
      if (!cancelled) {
        timer = setTimeout(poll, GPU_CONTENTION_POLL_MS);
      }
    };
    timer = setTimeout(poll, GPU_CONTENTION_POLL_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [contentionShown]);

  const toggleSidebar = () => {
    setCollapsed((prev) => !prev);
  };

  useEffect(() => {
    if (!window.fsAPI?.readImageAsDataURL) return;
    // undefined = not yet attempted; [] = attempted but file gone; [...] = loaded
    const needsRestore = messages.some(
      (m) => m.images === undefined && /\[image_path:[^\]]+\]/.test(m.content)
    );
    if (!needsRestore) return;
    let cancelled = false;
    (async () => {
      const restored = await Promise.all(
        messages.map(async (m) => {
          if (m.images !== undefined || !/\[image_path:[^\]]+\]/.test(m.content)) return m;
          // Marker paths are percent-encoded by the backend; decode before
          // touching the filesystem.
          const paths = getImagePaths(m.content);
          const images = (
            await Promise.all(
              paths.map((p) =>
                window.fsAPI.readImageAsDataURL(p).catch((error) => {
                  // The message renders without this image: degraded, named.
                  log.warn(`Could not reload the image attachment ${p}`, error);
                  return null;
                })
              )
            )
          ).filter(Boolean);
          return { ...m, images };
        })
      );
      if (!cancelled) setMessages(restored);
    })();
    return () => {
      cancelled = true;
    };
  }, [messages]);

  // Load the available models list. The header names the conversation's
  // model by looking its id up in this list, whichever of the two loads lands
  // first (#217).
  useEffect(() => {
    tracedFetch(`${API_BASE_URL}/llms/local`)
      .then((res) => res.json())
      .then((data) => {
        setModels(Array.isArray(data) ? data : []);
      })
      .catch((err) => {
        log.error("Failed to fetch models", err);
      })
      .finally(() => {
        setModelsLoaded(true);
      });
  }, []);

  const fetchMessagesAndConversations = useCallback(async () => {
    try {
      const [msgRes, convRes] = await Promise.all([
        tracedFetch(`${API_BASE_URL}/conversations/${id}/fetch_messages`),
        tracedFetch(`${API_BASE_URL}/conversations/`),
      ]);
      const msgs = await msgRes.json();
      const convs = await convRes.json();
      // The sidebar's list is the same whatever conversation is shown: it is
      // always refreshed (a first turn's new title and the new order).
      convs.sort((a, b) => new Date(b.last_message_time) - new Date(a.last_message_time));
      setConversations(convs);
      // The rows of a conversation the page has left (or a page that is
      // gone) land nowhere: they would replace the other conversation's.
      if (idRef.current !== id) return;
      // initialize starred state from backend
      const starredMap = {};
      msgs.forEach((m) => {
        if (m.starred) {
          starredMap[m.id] = true;
        }
      });
      setStarredIds(starredMap);
      setMessages((prev) => {
        const sameConv = loadedConvIdRef.current === id;
        const fetched = msgs.map((m, i) => ({
          ...m,
          images: (sameConv && prev[i]?.images) || m.images,
        }));
        return sameConv ? mergeFetchedMessages(prev, fetched) : fetched;
      });
      loadedConvIdRef.current = id;
    } catch (err) {
      log.error("Failed to fetch messages and conversations", err);
    }
  }, [id]);

  const handleAsk = useCallback(
    async (question, images = [], imagePaths = [], attachmentPaths = []) => {
      setLoading(true);
      isStreamingRef.current = true;

      // Sending is the user's return to the bottom: the view follows the
      // answer again, and a pending scroll re-check must not undo that.
      clearTimeout(scrollCheckTimerRef.current);
      userScrolledUpRef.current = false;

      const isFirstMessage = messagesLengthRef.current === 0;

      if (isFirstMessage) {
        setFirstReplyPending(true);
      }

      // `local: true` marks optimistic entries a refetch may not know about
      // yet, so mergeFetchedMessages keeps them instead of erasing them (#303).
      const userMessage = {
        id: Date.now(),
        sender: "user",
        content: question,
        images,
        // Names of the documents attached to this turn (#492): shown as chips
        // right away; after a reload they come back from the stored markers.
        attachmentNames: attachmentPaths.map(baseName),
        local: true,
      };

      const assistantMessage = {
        id: Date.now() + 1,
        sender: "llm",
        content: "",
        // Live NDJSON trace (#90): thinking / tool_call / tool_result events
        // collected as they stream, replayed by the TraceStrip above the bubble,
        // and the compaction markers (#614) rendered above the strip.
        // `streaming` gates the strip's expanded-while-only-content behavior.
        trace: [],
        streaming: true,
        local: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);

      try {
        // A change made just before sending (a slider still within its save
        // delay, a save on its way) is saved first, so this turn runs on it.
        await flush();
        const current = getValues();
        // The title runs BEFORE the chat, and the chat starts once the title
        // stream has ended (success, HTTP error or network failure alike).
        // llama.cpp runs one slot with no prompt cache, so a title that ran
        // after the first turn would overwrite the conversation's KV and the
        // second message would re-read the whole conversation. Both requests
        // already serialize on the backend's generation lock, so awaiting adds
        // only the title's own time (the prefill of the question, plus the
        // model load when the model is cold) before the chat request starts
        // its attachment and Knowledge Base preparation.
        if (isFirstMessage) {
          await (async () => {
            try {
              const titleRes = await tracedFetch(
                `${API_BASE_URL}/conversations/${id}/generate_title`,
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ question }),
                }
              );

              if (titleRes.ok) {
                const reader = titleRes.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let fullTitle = "";

                let titleDone = false;
                while (!titleDone) {
                  const { done, value } = await reader.read();
                  titleDone = done;
                  if (done) {
                    break;
                  }

                  const chunk = decoder.decode(value, { stream: true });
                  fullTitle += chunk;

                  const name = fullTitle.trim() || t("chat:conversation.untitled");
                  setConversations((prevConvs) =>
                    prevConvs.map((conv) => (conv.id === Number(id) ? { ...conv, name } : conv))
                  );
                }
              }
            } catch (err) {
              log.error("Title generation failed", err);
            }
          })();
        }

        // Stream response using fetch + ReadableStream
        const responseRes = await tracedFetch(`${API_BASE_URL}/conversations/${id}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            images,
            image_paths: imagePaths,
            attachments: attachmentPaths,
            // The conversation's values, once loaded (the backend uses the
            // stored ones otherwise); the instructions only reach the model
            // through the turn.
            ...(current.temperature === null ? {} : { temperature: current.temperature }),
            ...(current.top_p === null ? {} : { top_p: current.top_p }),
            ...(current.custom_prompt === null ? {} : { custom_prompt: current.custom_prompt }),
          }),
        });

        if (responseRes.ok) {
          // NDJSON reader (#90): the body is one JSON event per line
          // (application/x-ndjson). We split on "\n" and carry the trailing
          // partial line across reads. Answer events stream into the bubble
          // exactly as raw text did before; thinking / tool_call / tool_result
          // and compaction `done` accrue into the message's live trace;
          // compaction `start` is a transient state; error maps to the red bubble;
          // done is terminal. Unknown event types are ignored (forward compat).
          // NOTE: the title-gen reader above is a SEPARATE plain-text loop.
          const reader = responseRes.body.getReader();
          const decoder = new TextDecoder("utf-8");
          let buffer = ""; // partial-line carry across reads
          let answerText = ""; // accumulated answer bubble text
          const traceEvents = []; // ordered thinking / tool / compaction-done events
          let gotFirstEvent = false;
          let parseWarned = false;
          let sawError = false;
          let sawDone = false;
          let sawMemoryWarning = false;
          // The memory brake, the idle delay or the user's Unload button
          // (`reason`) unloaded the model and this turn reloads it
          // (`memory_reload`): shown in
          // the pending bubble until the model speaks (its first answer,
          // thinking or tool event), an error, or the end of the turn. Notices the backend streams as answer text
          // come BEFORE it, so they never end it. Never persisted.
          let reloading = false;
          // A compaction in the pending bubble (#614): "during" from its
          // `start` (typing indicator + "Summarizing earlier messages..."),
          // "after" from its `done` (the indicator alone: the separator
          // already says what was summarized, and the model still re-reads
          // the shortened conversation -- clearing it outright would leave a
          // blank bubble). The model speaking (its first answer, thinking or
          // tool event), an error, or the end of the turn ends either.
          let compacting = false;

          // Push the current accumulators onto the streaming assistant message.
          const flushMessage = () => {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessage.id
                  ? {
                      ...msg,
                      content: answerText,
                      trace: traceEvents.slice(),
                      reloading,
                      compacting,
                    }
                  : msg
              )
            );
          };

          const handleEvent = (evt) => {
            if (!evt || typeof evt !== "object") {
              return;
            }
            switch (evt.t) {
              case "memory_reload":
                // Why the model reloads picks the copy (RELOAD_COPY): "idle"
                // (the idle delay unloaded it), "manual" (the user did) or
                // "memory" (the memory brake did; also any other reason).
                reloading = Object.hasOwn(RELOAD_COPY, evt.reason) ? evt.reason : "memory";
                break;
              case "compaction":
                // `start` is the transient state; `done` is the marker the
                // trace keeps (persisted, replayed on reload).
                if (evt.phase === "start") {
                  reloading = false;
                  compacting = "during";
                } else if (evt.phase === "done") {
                  compacting = "after";
                  traceEvents.push(evt);
                }
                break;
              case "answer":
                reloading = false;
                compacting = false;
                answerText += evt.text || "";
                break;
              case "thinking":
              case "tool_call":
              case "tool_result":
                reloading = false;
                compacting = false;
                traceEvents.push(evt);
                break;
              case "error":
                // The wire error event replaces the old in-band sentinel; map it
                // back to the SAME red bubble the persisted sentinel produces
                // (getDisplayContent / bubbleClass detect the sentinel substring).
                // The backend error text is already sentinel-prefixed.
                sawError = true;
                reloading = false;
                compacting = false;
                // The backend has the traceback; this side records that the
                // turn failed, on which conversation, and the engine's reason.
                // A curated turn (a prompt that does not fit, a timeout, the
                // loop limit) is the app's own answer, not a defect: WARNING,
                // so it does not badge the bug counter. A generic one is ERROR.
                (evt.curated ? log.warn : log.error)(`Generation failed for conversation ${id}`, {
                  code: evt.code || null,
                  raw: evt.raw || null,
                });
                // Text streamed before the failure stays above the red line.
                answerText = joinFailure(
                  answerText,
                  evt.text || `${ERROR_SENTINEL} ${t("chat:errors.generationFailed")}`
                );
                if (evt.detail) {
                  traceEvents.push({
                    t: "error_detail",
                    text: evt.detail,
                    request_id: evt.request_id || null,
                  });
                }
                // The engine identified WHY it failed (a card the bundled CUDA
                // build cannot drive, a driver too old, VRAM exhaustion): raise
                // the dialog that offers the matching remedy. Untyped failures
                // stay a red bubble and nothing else, as they always have.
                // A command popup open meanwhile closes first: the dialog
                // takes the screen and the keyboard.
                if (evt.code) {
                  setOpenCommand(null);
                  setEngineFailure({ code: evt.code, raw: evt.raw });
                }
                break;
              case "memory_warning":
                // Statement about NOW (this machine, this load): page state
                // only — never appended to the answer text or the trace, so
                // it survives neither the copy path nor a reload.
                // The copy follows `reason` (`bound` / `pressure` /
                // `pressure_external` / `gpu_contention`); `bound` alone
                // quotes a size.
                sawMemoryWarning = true;
                setMemoryWarning({
                  reason: evt.reason,
                  // Conversation + loaded model: what the `bound` copy quotes.
                  footprintBytes: evt.footprint_bytes,
                });
                break;
              case "done":
                sawDone = true;
                reloading = false;
                compacting = false;
                break;
              default:
                // Unknown event type: ignore for forward compatibility.
                break;
            }
          };

          const processLine = (rawLine) => {
            const line = rawLine.trim();
            if (!line) {
              return;
            }
            let evt;
            try {
              evt = JSON.parse(line);
            } catch (parseError) {
              // The line is skipped and the stream goes on: degraded, once
              // per turn so a garbled stream cannot flood the log.
              if (!parseWarned) {
                parseWarned = true;
                log.warn(`Skipping an unparseable stream line (${line.length} chars)`, parseError);
              }
              return;
            }
            handleEvent(evt);
            if (!gotFirstEvent) {
              gotFirstEvent = true;
              setLoading(false);
              setFirstReplyPending(false);
            }
            flushMessage();
          };

          try {
            let responseDone = false;
            while (!responseDone && !sawDone) {
              const { done, value } = await reader.read();
              responseDone = done;
              if (done) {
                break;
              }

              buffer += decoder.decode(value, { stream: true });
              let newlineIdx;
              while (!sawDone && (newlineIdx = buffer.indexOf("\n")) !== -1) {
                const rawLine = buffer.slice(0, newlineIdx);
                buffer = buffer.slice(newlineIdx + 1);
                processLine(rawLine);
              }
            }
            // Defensive flush of any trailing line (NDJSON ends every line with
            // "\n", so this normally no-ops).
            if (!sawDone && buffer.trim()) {
              processLine(buffer);
              buffer = "";
            }
          } catch (streamError) {
            log.error("Streaming error during response generation", streamError);
            answerText = joinFailure(
              answerText,
              `${ERROR_SENTINEL} ${t("chat:errors.connectionInterrupted")}`
            );
            sawError = true;
            flushMessage();
          } finally {
            assistantMessage.content = answerText;
            // A completed turn that carried no memory warning clears the amber
            // block: the margin recovered (compaction sufficed, or the model
            // changed), so a stale warning must not linger.
            if (!sawMemoryWarning) {
              setMemoryWarning(null);
            }
            // Turn is over: stop the live strip (it settles to the collapsed
            // summary) and keep only the compactions and the failure's cause
            // of an error turn's trace to match persisted history (the backend
            // persists nothing else for error turns).
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessage.id
                  ? {
                      ...msg,
                      content: answerText,
                      trace: sawError ? errorTurnTrace(traceEvents) : traceEvents.slice(),
                      streaming: false,
                      reloading: false,
                      compacting: false,
                    }
                  : msg
              )
            );
          }
        } else {
          log.error("Server error during response generation", { status: responseRes.status });

          try {
            await tracedFetch(`${API_BASE_URL}/conversations/${id}/store_error_message`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
            });
          } catch (storeError) {
            log.error("Failed to store error message", storeError);
          }

          assistantMessage.content = `${ERROR_SENTINEL} ${t("chat:errors.generationFailed")}`;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessage.id
                ? {
                    ...msg,
                    content: assistantMessage.content,
                    trace: null,
                    streaming: false,
                    reloading: false,
                    compacting: false,
                  }
                : msg
            )
          );
        }
      } catch (err) {
        log.error("Failed to send message", err);
      }

      isStreamingRef.current = false;
      await fetchMessagesAndConversations();
      // The turn is over (the model it loaded is up): the header's Unload
      // button reads the loaded model again.
      setTurnsEnded((count) => count + 1);
      // The page moved to another conversation while this turn ran: its load
      // waited for the turn (it never overwrites a streaming answer), so it
      // runs now and the page shows what it names.
      if (idRef.current !== null && idRef.current !== id) {
        setLoadAttempt((n) => n + 1);
      }

      if (isFirstMessage) {
        setFirstReplyPending(false);
      }

      setLoading(false);
    },
    [id, fetchMessagesAndConversations, t, flush, getValues]
  );

  // Load conversation data when ID changes
  useEffect(() => {
    if (!id) {
      return;
    }

    // Don't overwrite in-flight streaming state with stale DB data
    if (isStreamingRef.current) {
      return;
    }

    // A load the page has moved on from (another conversation, a new run of
    // this effect) lands nowhere: its rows belong to what was shown then.
    let cancelled = false;

    const loadConversationData = async () => {
      // Re-check inside async in case streaming started between effect trigger and execution
      if (isStreamingRef.current) {
        return;
      }

      try {
        // A row older than a save or a /model made while it loads is dropped,
        // and so is a row of a conversation the page has left.
        const stamp = loadStamp();
        const [convRes, msgRes, convDetailRes] = await Promise.all([
          tracedFetch(`${API_BASE_URL}/conversations/`),
          tracedFetch(`${API_BASE_URL}/conversations/${id}/fetch_messages`),
          tracedFetch(`${API_BASE_URL}/conversations/${id}`),
        ]);

        // Load conversations
        const convs = await convRes.json();
        const msgs = await msgRes.json();
        if (cancelled) return;
        convs.sort((a, b) => new Date(b.last_message_time) - new Date(a.last_message_time));
        setConversations(convs);

        // Load messages
        const starredMap = {};
        msgs.forEach((m) => {
          if (m.starred) {
            starredMap[m.id] = true;
          }
        });
        setStarredIds(starredMap);
        setMessages((prev) => {
          const sameConv = loadedConvIdRef.current === id;
          const fetched = msgs.map((m, i) => ({
            ...m,
            images: (sameConv && prev[i]?.images) || m.images,
          }));
          return sameConv ? mergeFetchedMessages(prev, fetched) : fetched;
        });
        loadedConvIdRef.current = id;

        // Load the conversation's model and settings. `max_tokens` is
        // deliberately not read: the output budget is computed backend-side
        // from the running context window. llm_id can be null after the
        // model was deleted (conversations survive, no auto-fallback, #225);
        // the orphan check below decides whether to block the composer.
        if (!convDetailRes.ok) {
          throw new Error(`Conversation ${id} could not be read (HTTP ${convDetailRes.status})`);
        }
        const conversation = await convDetailRes.json();
        if (cancelled) return;
        adopt(conversation, stamp);
        setLoadFailed(false);
      } catch (err) {
        log.error("Failed to fetch conversations", err);
        if (cancelled) return;
        setLoadFailed(true);
      }

      // The first message from the new-conversation page: it runs on the
      // values of the conversation as created, adopted just above.
      if (location.state && location.state.initialQuestion && !initialHandled) {
        setInitialHandled(true);
        await handleAsk(
          location.state.initialQuestion,
          location.state.initialImages || [],
          location.state.initialImagePaths || [],
          location.state.initialAttachments || []
        );
        // The question is consumed from the history entry, but only while
        // the page still shows this conversation: after a switch during the
        // first turn, the current entry is the other conversation's, and
        // replacing it with this path would bring the user back here.
        if (idRef.current === id) {
          navigate(location.pathname, { replace: true, state: {} });
        }
      } else if (!location.state || !location.state.initialQuestion) {
        try {
          const msgs = await apiClient.get(`/conversations/${id}/fetch_messages`);
          if (cancelled) return;
          // initialize starred state on initial load
          const starredMap = {};
          msgs.forEach((m) => {
            if (m.starred) {
              starredMap[m.id] = true;
            }
          });
          setStarredIds(starredMap);
          setMessages((prev) => {
            const sameConv = loadedConvIdRef.current === id;
            const imagesByContent = sameConv
              ? new Map(prev.filter((m) => m.images?.length > 0).map((m) => [m.content, m.images]))
              : new Map();
            const fetched = msgs.map((m) => ({
              ...m,
              images: imagesByContent.get(m.content) || m.images,
            }));
            return sameConv ? mergeFetchedMessages(prev, fetched) : fetched;
          });
          loadedConvIdRef.current = id;
        } catch (err) {
          log.error("Failed to fetch messages", err);
        }
      }
    };

    loadConversationData();
    return () => {
      cancelled = true;
    };
  }, [
    id,
    location.state,
    handleAsk,
    adopt,
    loadStamp,
    navigate,
    location.pathname,
    initialHandled,
    loadAttempt,
  ]);

  const retryLoad = () => {
    setLoadFailed(false);
    setLoadAttempt((n) => n + 1);
  };

  // Detect when user manually scrolls
  useEffect(() => {
    const scrollContainer = scrollRef.current;
    if (!scrollContainer) {
      return;
    }

    let lastScrollTop = scrollContainer.scrollTop;

    const handleScroll = () => {
      const currentScrollTop = scrollContainer.scrollTop;

      // If user scrolled up (not down), immediately stop auto-scrolling
      if (currentScrollTop < lastScrollTop) {
        userScrolledUpRef.current = true;
      }

      lastScrollTop = currentScrollTop;

      // Clear any pending timeout
      clearTimeout(scrollCheckTimerRef.current);

      // Wait a bit before checking position to avoid false positives during auto-scroll
      scrollCheckTimerRef.current = setTimeout(() => {
        const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
        const isAtBottom = scrollHeight - scrollTop - clientHeight < 100; // Increased threshold to 100px

        // Only mark as scrolled up if user is significantly away from bottom
        userScrolledUpRef.current = !isAtBottom;
      }, 100); // Small delay to debounce
    };

    scrollContainer.addEventListener("scroll", handleScroll);
    return () => {
      clearTimeout(scrollCheckTimerRef.current);
      scrollContainer.removeEventListener("scroll", handleScroll);
    };
  }, []);

  // A /context panel or the /compact status row the user just asked for
  // shows at the end of the thread: bring it into view. Only when the thread
  // grows: closing a panel never moves the view.
  const scrolledForRef = useRef({ panels: 0, compacting: false });
  useEffect(() => {
    const previous = scrolledForRef.current;
    const grew = contextPanels.length > previous.panels || (compactingHere && !previous.compacting);
    scrolledForRef.current = { panels: contextPanels.length, compacting: compactingHere };
    if (grew && scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [contextPanels.length, compactingHere]);

  // A /compact this page did not start (one started before the user left
  // the page and came back) re-reads the conversation when it ends, so its
  // separator shows.
  const rereadRef = useRef(fetchMessagesAndConversations);
  useLayoutEffect(() => {
    rereadRef.current = fetchMessagesAndConversations;
  });
  useEffect(() => {
    return subscribeCompacting((conversationId) => {
      if (
        conversationId === String(idRef.current) &&
        !isCompacting(conversationId) &&
        !startedHereRef.current.has(conversationId)
      ) {
        rereadRef.current?.();
      }
    });
  }, []);

  // Auto-scroll only when streaming is done or user hasn't scrolled up
  useEffect(() => {
    // Only auto-scroll if user hasn't manually scrolled up
    // OR if loading just finished (scroll once to final position)
    if (scrollRef.current && !userScrolledUpRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  const handleConversationClick = (newId) => navigate(conversationPath(newId));

  const handleRename = (cid, newName) =>
    setConversations((prev) => prev.map((c) => (c.id === cid ? { ...c, name: newName } : c)));

  const handleDelete = async (cid) => {
    // Ownership contract (#228): ChatCollapsibleSection owns the DELETE request.
    // This parent handler runs AFTER the child has deleted and does post-delete
    // UI only — optimistic filter, server refresh, and navigating away from a
    // now-deleted open conversation. It must NOT issue its own DELETE, or the
    // request fires twice (~10ms apart, distinct request ids).
    setConversations((prev) => prev.filter((conv) => conv.id !== cid));
    await fetchMessagesAndConversations();
    if (cid === Number(id)) {
      navigate("/erudi/chat");
    }
  };

  // Toggle star state optimistically, then reconcile with the server.
  const toggleStar = async (msgId) => {
    const isStarred = starredIds[msgId];
    const url = `${API_BASE_URL}/conversations/${isStarred ? "unstar_message" : "star_message"}`;
    // Optimistic: flip immediately so the star responds on click.
    setStarredIds((prev) => ({ ...prev, [msgId]: !isStarred }));
    try {
      const res = await tracedFetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message_id: msgId }),
      });
      // fetch only rejects on network errors — a non-ok status is also a failure.
      if (!res.ok) {
        throw new Error(`star toggle failed with status ${res.status}`);
      }
    } catch (err) {
      // Rollback must stay SILENT (owner decision, #136): no toast, no modal —
      // a star is low-stakes, so the only trace is this debug-level log line.
      setStarredIds((prev) => ({ ...prev, [msgId]: isStarred }));
      log.debug("Star toggle failed, reverting optimistic update", err);
    }
  };

  // Orphaned conversation (#225): its assigned model was deleted (llm_id null)
  // or no longer exists in the installed list. Sending is blocked and the
  // header turns red until the user switches to an installed model with
  // /model (the header panel's Model button stays usable). An assigned model
  // whose weights are gone (an orphaned KB assistant, weights_available ===
  // false) can only fail, so it blocks the same way; undefined/null
  // weights_available (remote or older backend) never blocks.
  const llmId = conversationSettings.values.llm_id;
  const assignedModel = models.find((m) => m.id === llmId);
  const isModelOrphaned =
    conversationSettings.loaded &&
    modelsLoaded &&
    (llmId === null || !assignedModel || assignedModel.weights_available === false);

  const errorNotice = (key) => ({ tone: "error", text: t(key) });

  // /context (#608): GET the usage of the next request and draw it at the end
  // of the thread. Never loads the model. An answer for a conversation the
  // page has left, or read before a send or a re-read since, is dropped. A
  // conversation whose model is gone is loaded again (its header then asks
  // for a model) and told to pick one.
  const showContextUsage = async () => {
    const conversationId = id;
    const epoch = contextEpochRef.current;
    const current = () => idRef.current === conversationId && contextEpochRef.current === epoch;
    try {
      const usage = await apiClient.get(`/conversations/${conversationId}/context`);
      if (!current()) return;
      contextPanelSeqRef.current += 1;
      const key = contextPanelSeqRef.current;
      setContextPanels((prev) => [...prev, { key, usage }]);
    } catch (err) {
      // The API client wrote the request's record at the level of its status.
      if (!current()) return;
      if (err?.data?.error?.detail?.reason === "no_model") {
        setLoadAttempt((n) => n + 1);
        setCommandNotice(errorNotice("chat:commands.pickModelFirst"));
      } else {
        setCommandNotice(errorNotice("chat:context.failed"));
      }
    }
  };

  // /compact (#585), once confirmed. One run per conversation (a second
  // confirmation while it runs starts nothing), held in the session's store
  // so the conversation stays locked, its status row shown, wherever the
  // user goes meanwhile. POST through tracedFetch: no 30 s client timeout (a
  // summary can outlast it), a ceiling of 10 minutes instead. What follows
  // the answer (`compactOutcome`) happens only while the page still shows
  // this conversation, checked again after every wait.
  const compactConversation = async () => {
    const conversationId = id;
    if (!startCompacting(conversationId)) return;
    startedHereRef.current.add(conversationId);
    const here = () => idRef.current === conversationId;
    const controller = new AbortController();
    const ceiling = setTimeout(() => controller.abort(), COMPACT_CEILING_MS);
    let outcome;
    try {
      const res = await tracedFetch(`${API_BASE_URL}/conversations/${conversationId}/compact`, {
        method: "POST",
        signal: controller.signal,
      });
      let body = null;
      try {
        body = await res.json();
      } catch {
        // No JSON body: the outcome is unknown; compactOutcome says so.
      }
      outcome = compactOutcome(res.ok, body);
      if (outcome.notice?.tone === "error") {
        // The backend refused (4xx, its own INFO record) or failed (its WARNING
        // or ERROR record); this side records that the page told the user.
        const write = res.status >= 500 ? log.warn : log.info;
        write(`Manual compaction of conversation ${conversationId} ended in HTTP ${res.status}`, {
          type: body?.error?.type ?? null,
          reason: body?.error?.detail?.reason ?? null,
        });
      }
    } catch {
      if (controller.signal.aborted) {
        log.warn(
          `Manual compaction of conversation ${conversationId} had no answer after ${COMPACT_CEILING_MS / 60000} min; gave up`
        );
      }
      // Otherwise tracedFetch wrote the request's api.failure record.
      outcome = {
        notice: { tone: "error", key: "chat:compact.errors.unknown" },
        after: AFTER_COMPACT.REREAD,
      };
    } finally {
      clearTimeout(ceiling);
    }
    try {
      if (!here()) return;
      const notice = outcome.notice
        ? { tone: outcome.notice.tone, text: t(outcome.notice.key, outcome.notice.values) }
        : null;
      if (outcome.after === AFTER_COMPACT.LEAVE) {
        // The conversation is gone: no re-read (it would only fail); the
        // new-conversation page shows the notice and the current list.
        navigate("/erudi/chat", { state: { notice } });
        return;
      }
      if (outcome.after === AFTER_COMPACT.RELOAD) {
        setLoadAttempt((n) => n + 1);
      } else if (outcome.after === AFTER_COMPACT.REREAD) {
        await withinCeiling(fetchMessagesAndConversations(), COMPACT_REREAD_CEILING_MS);
        if (!here()) return;
        clearContextPanels();
      }
      setCommandNotice(notice);
    } finally {
      // Ended first, then forgotten as this page's own run: the page's
      // listener must not take it for another page's run and re-read again.
      finishCompacting(conversationId);
      startedHereRef.current.delete(conversationId);
    }
  };

  // A command from the composer or the header panel. /context draws its
  // panel; every other one opens its popup (/compact: its confirmation). A
  // conversation without a usable model is told to pick one for /compact and
  // /context, which would only fail.
  const runCommand = (command) => {
    setCommandNotice(null);
    if ((command.name === "compact" || command.name === "context") && isModelOrphaned) {
      setCommandNotice(errorNotice("chat:commands.pickModelFirst"));
      return;
    }
    if (command.name === "context") {
      showContextUsage();
      return;
    }
    setOpenCommand(command);
  };

  // A sent message makes the /context snapshots stale, and drops any still
  // on its way.
  const sendMessage = async (...args) => {
    const conversationId = id;
    clearContextPanels();
    setCommandNotice(null);
    await handleAsk(...args);
    // Context is usable again once the answer starts (like Model): a
    // snapshot taken while it streamed describes a request already sent.
    if (idRef.current === conversationId) clearContextPanels();
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        disabled={loading}
        showCollapsible={true}
        onToggleSidebar={toggleSidebar}
        collapsed={collapsed}
      />

      <aside
        className={`relative bg-[#272727] text-white transition-all duration-300 ease-in-out ${
          collapsed ? "w-0 p-0" : "w-80 p-6 flex flex-col"
        }`}
      >
        {!collapsed && (
          <>
            <h1 className="text-3xl font-bold mb-6 flex-shrink-0">{t("chat:history.title")}</h1>
            {/*<ChatCollapsibleSection title="Hot Chats"
              disabled={loading}
            />} coming in next version*/}
            <div className="flex-1 mb-4 overflow-hidden">
              <ChatCollapsibleSection
                title={t("chat:history.previousChats")}
                items={conversations}
                selectedId={Number(id)}
                onSelect={handleConversationClick}
                onRename={handleRename}
                onDelete={handleDelete}
                disabled={loading}
              />
            </div>
          </>
        )}
      </aside>
      <main className="flex-1 flex flex-col bg-gradient-to-br from-[#041915] to-[#0f2d27] overflow-hidden">
        <div className="relative flex justify-center w-full px-8 pt-6">
          <ConversationHeader
            modelName={assignedModel?.name ?? ""}
            attention={isModelOrphaned}
            commands={COMMANDS}
            disabledCommands={
              compactingHere ? ALL_COMMAND_NAMES : loading ? COMMANDS_DISABLED_WHILE_ANSWERING : []
            }
            onCommand={runCommand}
            saveFailed={conversationSettings.saveFailed}
          >
            {/* Re-read the loaded model when a turn has completed (turnsEnded),
                when one starts streaming (loading) and when the model changes. */}
            <UnloadModelButton refreshKey={`${turnsEnded}:${loading}:${llmId}`} />
          </ConversationHeader>
        </div>

        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-10 pt-10 pb-4"
          style={{
            scrollbarWidth: "none",
            msOverflowStyle: "none",
          }}
        >
          <style>{HIDE_SCROLLBAR_CSS}</style>
          <div className="flex flex-col gap-6">
            {messages.map((msg) => {
              const isUser = msg.sender === "user";
              const isError = !isUser && msg.content.includes("[ERROR_MESSAGE_SYSTEM]");
              const alignmentClass = alignmentClassFor(isUser);
              const bubbleClass = bubbleClassFor(isUser, isError);

              // Reasoning/trace strip (#90): only for assistant turns that carried
              // thinking or tool activity, never for error turns (their persisted
              // trace holds compactions and the failure's cause only). `traceLive` is true while the strip
              // is the only content (streaming, no answer yet) -> expanded; it
              // flips false on the first answer event -> auto-collapse.
              const traceEvents = !isUser ? msg.trace : null;
              const showTrace = !isUser && !isError && hasStripEvents(traceEvents);
              const traceLive = showTrace && !!msg.streaming && !msg.content;
              // Where a compaction took over (#614): separators above the
              // strip and the bubble, error turns included; a /compact run
              // saved on this answer (#585) below it.
              const compactions = !isUser ? automaticCompactions(traceEvents) : [];
              const manualRuns = !isUser ? manualCompactions(traceEvents) : [];

              // Show TypingIndicator for assistant messages that are loading, but
              // not once the live trace strip has taken over the pre-answer wait.
              const showTypingIndicator =
                !isUser &&
                loading &&
                !msg.content &&
                !showTrace &&
                !msg.reloading &&
                !msg.compacting;
              // The model is being reloaded after the memory brake or the idle
              // timeout unloaded it: its own state, independent of the content (a notice the
              // backend streamed as answer text may already be there).
              const showReload = !isUser && !!msg.reloading;
              // A compaction is pending in the turn: the same slot, whatever
              // the strip shows (a later model call of the turn can compact).
              // Its text only while the summary is being written.
              const showCompacting = !isUser && !!msg.compacting;
              const showCompactingText = showCompacting && msg.compacting === "during";

              // Attached images (this session) render in their OWN panel,
              // separate from the text bubble. `displayText` is the readable
              // text with internal attachment markers stripped.
              const hasImages = isUser && msg.images?.length > 0;
              // Attached documents (#492): the live turn carries their names,
              // a reloaded one recovers them from the stored [file_path:...]
              // markers. Either way only names are shown - the extracted text
              // was a one-turn ingredient, never persisted content.
              const attachmentNames = isUser
                ? msg.attachmentNames?.length
                  ? msg.attachmentNames
                  : getAttachmentNames(msg.content)
                : [];
              const hasAttachments = attachmentNames.length > 0;
              // An unexpected failure (#615): its headline is the interface
              // language's, whatever the stored English line; its trace holds
              // the cleaned cause behind Details. A turn stored before it
              // existed has the headline only.
              const failureDetail = isError ? errorDetailOf(msg.trace) : null;
              const failureHeadline =
                isError && isGenericFailure(msg.content, msg.trace)
                  ? t("chat:errors.generationFailed")
                  : undefined;
              const displayText = getDisplayContent(msg.content, failureHeadline);

              return (
                <div key={msg.id} className={`group flex flex-col mb-2 ${alignmentClass}`}>
                  {compactions.length > 0 && <CompactionMarker events={compactions} />}
                  {showTrace && <TraceStrip events={traceEvents} live={traceLive} />}
                  {/* Attached images: their OWN glass panel (chat-header
                      material), a sibling of the text bubble, not nested in it. */}
                  {hasImages && (
                    <div
                      className={[
                        "mb-2 w-fit max-w-[75%] rounded-2xl p-2",
                        "border border-white/10",
                        "bg-[rgba(22,40,36,0.45)] backdrop-blur-[18px] saturate-[1.4]",
                      ].join(" ")}
                    >
                      <div className="flex flex-wrap gap-2">
                        {msg.images.map((src, i) => (
                          <img
                            key={i}
                            src={src}
                            alt={t("chat:message.attachmentAlt", { index: i + 1 })}
                            className="max-h-64 max-w-full rounded-xl border border-white/10"
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Attached documents: a chip per file, above the bubble. */}
                  {hasAttachments && (
                    <div className="mb-2 flex w-fit max-w-[75%] flex-wrap gap-2">
                      {attachmentNames.map((name, i) => (
                        <span
                          key={`${name}-${i}`}
                          className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--line)] px-2 py-0.5 text-xs text-[var(--ink-faint)]"
                        >
                          <FileText className="h-3.5 w-3.5 shrink-0" />
                          <span className="truncate max-w-[16rem]">{name}</span>
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Text bubble. Skipped for an attachment-only user message so
                      no empty bubble shows beneath the picture or the chips. */}
                  {((!hasImages && !hasAttachments) || displayText || showTypingIndicator) && (
                    <div
                      className={`break-words w-fit max-w-[75%] p-4 rounded-2xl overflow-wrap break-word ${bubbleClass}`}
                    >
                      {showTypingIndicator ? (
                        <div className="flex flex-col gap-2">
                          <div className="flex items-start pt-1">
                            <TypingIndicator size={8} className="-mt-1" />
                          </div>
                          {firstReplyPending && (
                            <div className="text-xs text-gray-400 italic mt-1">
                              {t("chat:message.firstReplyHint")}
                            </div>
                          )}
                        </div>
                      ) : isUser || msg.content.includes("[ERROR_MESSAGE_SYSTEM]") ? (
                        // Keep user messages and error messages as plain text.
                        <div className="flex flex-col gap-2">
                          {/* Reloaded sessions without image bytes: placeholder
                              markers, only when there are no real images above. */}
                          {!hasImages &&
                            (() => {
                              const fallbackCount = (
                                msg.content.match(/\[image\]|\[image_path:[^\]]*\]/g) || []
                              ).length;
                              return Array.from({ length: fallbackCount }, (_, i) => (
                                <span
                                  key={i}
                                  className="inline-flex items-center gap-1 text-xs text-[var(--ink-faint)] border border-[var(--line)] rounded px-2 py-0.5 w-fit"
                                >
                                  {t("chat:message.imagePlaceholder")}
                                </span>
                              ));
                            })()}
                          {displayText && (
                            <pre className="whitespace-pre-wrap font-sans">{displayText}</pre>
                          )}
                          {failureDetail && <ErrorDetails detail={failureDetail} />}
                        </div>
                      ) : (
                        // Assistant normal messages: render markdown
                        (msg.content || (!showReload && !showCompacting)) && (
                          <MarkdownRenderer content={msg.content} />
                        )
                      )}
                      {showReload && (
                        <div className="flex flex-col gap-2">
                          <div className="flex items-start pt-1">
                            <TypingIndicator size={8} className="-mt-1" />
                          </div>
                          <div className="text-xs text-gray-400 italic mt-1">
                            {t(RELOAD_COPY[msg.reloading])}
                          </div>
                        </div>
                      )}
                      {showCompacting && (
                        <div className="flex flex-col gap-2">
                          <div className="flex items-start pt-1">
                            <TypingIndicator size={8} className="-mt-1" />
                          </div>
                          {showCompactingText && (
                            <div className="text-xs text-gray-400 italic mt-1">
                              {t("chat:message.compacting")}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                  <div className="flex mt-1 space-x-2 opacity-0 group-hover:opacity-100">
                    {/* Copy button */}
                    <button
                      onClick={() => {
                        // Copy the readable text the user sees, not the raw
                        // stored content with internal attachment markers.
                        navigator.clipboard
                          .writeText(getDisplayContent(msg.content, failureHeadline))
                          .then(() => {
                            setCopiedMessageId(msg.id);
                            clearTimeout(copiedTimerRef.current);
                            copiedTimerRef.current = setTimeout(
                              () => setCopiedMessageId(null),
                              1000
                            );
                          });
                      }}
                      className="text-gray-400 hover:text-white transition-colors"
                      title={t("chat:message.copy")}
                    >
                      {copiedMessageId === msg.id ? (
                        <Check size={16} className="text-green-400" />
                      ) : (
                        <Copy size={16} />
                      )}
                    </button>
                    {/* Star button */}
                    <button
                      onClick={() => toggleStar(msg.id)}
                      className="text-gray-400 hover:text-white transition-colors"
                      title={t("chat:message.star")}
                    >
                      <Star
                        size={16}
                        className={starredIds[msg.id] ? "text-yellow-400" : ""}
                        fill={starredIds[msg.id] ? "currentColor" : "none"}
                      />
                    </button>
                  </div>
                  {manualRuns.length > 0 && <CompactionMarker events={manualRuns} below />}
                </div>
              );
            })}
            {contextPanels.map((panel) => (
              <ContextUsagePanel
                key={panel.key}
                usage={panel.usage}
                onDismiss={() =>
                  setContextPanels((prev) => prev.filter((p) => p.key !== panel.key))
                }
              />
            ))}
            {compactingHere && (
              <div
                role="status"
                className="flex items-center justify-center gap-2 text-xs italic text-gray-400"
              >
                <TypingIndicator size={6} />
                <span>{t("chat:message.compacting")}</span>
              </div>
            )}
          </div>
        </div>
        <div className="sticky bottom-0 left-0 right-0 px-10 py-10 backdrop-blur-md flex justify-center w-full">
          <div className="w-full max-w-lg">
            {/* Amber memory warning, by reason, one at a time (the backend's
                precedence: a brake notice, then `gpu_contention`, then
                `bound`): the computer's memory bounds this conversation and
                even after auto-compaction the next request nearly fills that
                bound (`bound`); the computer is under memory pressure and the
                conversation will be summarized sooner (`pressure`); the
                pressure comes from other apps (`pressure_external`); or
                another app is taking the graphics card's memory and answers
                are slower (`gpu_contention`, polled away once it ends). An
                unknown reason shows nothing. Same amber vocabulary as the
                DragDropArea rejection toast. */}
            {memoryWarning && Object.hasOwn(MEMORY_WARNING_KEYS, memoryWarning.reason) && (
              <div
                role="alert"
                className="mb-3 flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs"
              >
                <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                <span>
                  {t(MEMORY_WARNING_KEYS[memoryWarning.reason], {
                    size: formatBytes(memoryWarning.footprintBytes) ?? "?",
                  })}
                </span>
              </div>
            )}
            {loadFailed && (
              <div
                role="alert"
                className="mb-3 flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-200 text-xs"
              >
                <span>{t("chat:conversation.loadFailed")}</span>
                <button
                  type="button"
                  onClick={retryLoad}
                  className="shrink-0 rounded-md px-2.5 py-1 font-semibold bg-white/10 hover:bg-white/15 border border-white/20 text-gray-100 transition"
                >
                  {t("common:actions.retry")}
                </button>
              </div>
            )}
            <CommandNotice notice={commandNotice} onDismiss={() => setCommandNotice(null)} />
            <QuestionInput
              onSend={sendMessage}
              // Locked until the conversation's values are loaded: a turn
              // carries its instructions and sampling. Locked while /compact
              // runs too, its commands with it.
              disabled={
                loading || compactingHere || isModelOrphaned || !conversationSettings.loaded
              }
              canAttachImages={canAttachImages(assignedModel)}
              maxImages={maxImagesForModel(assignedModel)}
              commands={COMMANDS}
              onCommand={runCommand}
            />
          </div>
        </div>
      </main>

      {/* The popup of the command being run (/config, /effort, /model, and
          /compact's confirmation). */}
      <CommandPopups
        command={openCommand}
        onClose={() => setOpenCommand(null)}
        settings={conversationSettings}
        model={assignedModel}
        models={models}
        onPickModel={(model) => changeModel(model.id)}
        onCompact={compactConversation}
      />

      {/* The turn failed for a reason the app can act on. */}
      <EngineFailureModal notice={engineFailure} onDismiss={() => setEngineFailure(null)} />
    </div>
  );
}
