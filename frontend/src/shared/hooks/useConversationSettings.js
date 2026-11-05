import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import apiClient from "../../services/api/client";
import { createLogger } from "../../utils/logger";
import { defaultsFor } from "../../utils/samplingDefaults";
import { DEFAULT_REASONING_EFFORT } from "../../utils/reasoningEffort";
import { DEFAULT_COMPACTION_THRESHOLD } from "../../utils/compactionThreshold";

const log = createLogger("ConversationSettings");

/** Pass as the conversation id on the new-conversation page. */
export const DRAFT = "draft";

/** The six values `/config` shows and `/config global` saves, by API name. */
export const SETTING_FIELDS = Object.freeze([
  "reasoning_effort",
  "temperature",
  "top_p",
  "compaction_threshold",
  "web_search_enabled",
  "custom_prompt",
]);

const SAMPLING_FIELDS = new Set(["temperature", "top_p"]);
// The sliders are saved once they stop moving for this long.
export const SAMPLING_SAVE_DELAY_MS = 300;
// `source` of a value saved for the model with /config global.
const SOURCE_SAVED = "saved";

const EMPTY = Object.freeze({
  llm_id: null,
  reasoning_effort: null,
  temperature: null,
  top_p: null,
  compaction_threshold: null,
  web_search_enabled: null,
  custom_prompt: null,
});

// What the draft shows when the picked model's defaults cannot be read: its
// recommended sampling and the shipped defaults. Display only: a value the
// user did not set is still left to the backend at creation.
const unreadDefaults = (model) => {
  const sampling = defaultsFor(model);
  return {
    reasoning_effort: DEFAULT_REASONING_EFFORT,
    temperature: sampling.temperature,
    top_p: sampling.topP,
    compaction_threshold: DEFAULT_COMPACTION_THRESHOLD,
    web_search_enabled: false,
    custom_prompt: "",
  };
};

const fromConversation = (conversation) => ({
  llm_id: conversation.llm_id ?? null,
  reasoning_effort: conversation.reasoning_effort ?? DEFAULT_REASONING_EFFORT,
  temperature: conversation.temperature ?? null,
  top_p: conversation.top_p ?? null,
  compaction_threshold: conversation.compaction_threshold ?? DEFAULT_COMPACTION_THRESHOLD,
  web_search_enabled: Boolean(conversation.web_search_enabled),
  custom_prompt: conversation.custom_prompt ?? "",
});

/**
 * The one owner of a conversation's settings (#609): the header panel,
 * /config, /config global, /effort and /model all read and write through it,
 * so two views can never disagree.
 *
 * In a conversation (`conversationId` a real id) every change is saved with a
 * one-field PATCH: effort, web search, threshold and instructions at once,
 * creativity and diversity 300 ms after the slider stops (they also ride in
 * each turn's payload). Saves go out one after another, in the order they
 * were made; a failed one puts its control back to the last saved value
 * (unless a later change of the same control is on its way) and raises
 * `saveFailed`. `flush()` sends what is pending and resolves once every save
 * has settled: the send path and /model wait for it. `adopt(conversation)`
 * takes the values of a conversation row; `changeModel(llmId)` sends only
 * `{llm_id}` and adopts the conversation the backend returns, with the new
 * model's values applied server-side.
 *
 * On the new-conversation page (`DRAFT`) nothing is saved: `selectModel(model)`
 * reads that model's resolved defaults (`GET /llms/{id}/conversation_defaults`),
 * a late answer for a previous model is ignored, and the values the user set
 * are remembered. On a model switch, the values the user did not set take the
 * new model's; those set follow the /model rule (the new model's saved values
 * and its creativity and diversity replace them, the others stay).
 * `createPayload()` is the creation body: the model and the values the user
 * set, every other one resolved by the backend.
 *
 * `saveAsDefaults()` (/config global) saves the six current values for the
 * current model; empty instructions clear the instructions saved for it.
 */
export default function useConversationSettings(conversationId) {
  const isDraft = conversationId === DRAFT;
  const [values, setValues] = useState(EMPTY);
  const [loaded, setLoaded] = useState(false);
  const [saveFailed, setSaveFailed] = useState(false);

  // Synchronous mirrors: the send path reads the values right after adopt().
  const valuesRef = useRef(EMPTY);
  const idRef = useRef(conversationId);
  idRef.current = conversationId;

  // Conversation: last saved value per field, the latest save per field,
  // saves in flight per field, the save chain, the pending slider save.
  const savedRef = useRef({});
  const latestSaveRef = useRef({});
  const inFlightRef = useRef({});
  const saveSeqRef = useRef(0);
  const chainRef = useRef(Promise.resolve());
  const pendingSamplingRef = useRef({ id: null, patch: {} });
  const timerRef = useRef(null);

  // Draft: the fields the user set, and the current defaults request.
  const touchedRef = useRef(new Set());
  const requestRef = useRef(0);

  const write = useCallback((patch) => {
    valuesRef.current = { ...valuesRef.current, ...patch };
    setValues(valuesRef.current);
  }, []);

  const enqueuePatch = useCallback(
    (id, patch) => {
      const seq = ++saveSeqRef.current;
      const fields = Object.keys(patch);
      for (const field of fields) {
        latestSaveRef.current[field] = seq;
        inFlightRef.current[field] = (inFlightRef.current[field] ?? 0) + 1;
      }
      const run = async () => {
        try {
          await apiClient.patch(`/conversations/${id}`, patch);
          if (id !== idRef.current) return;
          for (const field of fields) savedRef.current[field] = patch[field];
          setSaveFailed(false);
        } catch (error) {
          log.error(`Failed to save ${fields.join(", ")} of conversation ${id}`, error);
          if (id !== idRef.current) return;
          const revert = {};
          for (const field of fields) {
            const superseded =
              latestSaveRef.current[field] !== seq ||
              Object.hasOwn(pendingSamplingRef.current.patch, field);
            if (!superseded) revert[field] = savedRef.current[field];
          }
          write(revert);
          setSaveFailed(true);
        } finally {
          for (const field of fields) inFlightRef.current[field] -= 1;
        }
      };
      chainRef.current = chainRef.current.then(run);
      return chainRef.current;
    },
    [write]
  );

  const sendPendingSampling = useCallback(() => {
    clearTimeout(timerRef.current);
    timerRef.current = null;
    const { id, patch } = pendingSamplingRef.current;
    pendingSamplingRef.current = { id: null, patch: {} };
    if (Object.keys(patch).length > 0) enqueuePatch(id, patch);
  }, [enqueuePatch]);

  // A new conversation (or the page leaving it): the slider save still pending
  // goes to the conversation it was made in, then the state starts over.
  useEffect(() => {
    valuesRef.current = EMPTY;
    setValues(EMPTY);
    setLoaded(false);
    setSaveFailed(false);
    savedRef.current = {};
    touchedRef.current = new Set();
    requestRef.current += 1;
    return () => sendPendingSampling();
  }, [conversationId, sendPendingSampling]);

  const flush = useCallback(() => {
    sendPendingSampling();
    return chainRef.current;
  }, [sendPendingSampling]);

  const setField = useCallback(
    (field, value) => {
      write({ [field]: value });
      if (idRef.current === DRAFT) {
        touchedRef.current.add(field);
        return;
      }
      if (SAMPLING_FIELDS.has(field)) {
        const pending = pendingSamplingRef.current;
        pendingSamplingRef.current = {
          id: idRef.current,
          patch: { ...pending.patch, [field]: value },
        };
        clearTimeout(timerRef.current);
        timerRef.current = setTimeout(sendPendingSampling, SAMPLING_SAVE_DELAY_MS);
        return;
      }
      enqueuePatch(idRef.current, { [field]: value });
    },
    [write, enqueuePatch, sendPendingSampling]
  );

  // A conversation row: every value, except one the user is still saving (a
  // reload of the row must not undo a change on its way).
  const adopt = useCallback(
    (conversation) => {
      const next = fromConversation(conversation);
      const busy = (field) =>
        (inFlightRef.current[field] ?? 0) > 0 ||
        Object.hasOwn(pendingSamplingRef.current.patch, field);
      const patch = {};
      for (const [field, value] of Object.entries(next)) {
        savedRef.current[field] = value;
        if (!busy(field)) patch[field] = value;
      }
      write(patch);
      setLoaded(true);
    },
    [write]
  );

  const changeModel = useCallback(
    async (llmId) => {
      sendPendingSampling();
      const id = idRef.current;
      let switched = null;
      chainRef.current = chainRef.current.then(async () => {
        try {
          switched = await apiClient.patch(`/conversations/${id}`, { llm_id: llmId });
        } catch (error) {
          log.error(`Failed to switch conversation ${id} to model ${llmId}`, error);
          if (id === idRef.current) setSaveFailed(true);
          return;
        }
        if (id === idRef.current) {
          adopt(switched);
          setSaveFailed(false);
        }
      });
      await chainRef.current;
      return switched;
    },
    [adopt, sendPendingSampling]
  );

  const selectModel = useCallback(
    (model) => {
      const previous = valuesRef.current.llm_id;
      if (!model || model.id === previous) return Promise.resolve();
      const isSwitch = previous !== null;
      const request = ++requestRef.current;
      write({ llm_id: model.id });
      setLoaded(false);

      // `incoming` holds the new model's value and source per field.
      const apply = (incoming) => {
        const patch = {};
        for (const field of SETTING_FIELDS) {
          const entry = incoming[field];
          if (!entry) continue;
          if (!touchedRef.current.has(field)) {
            patch[field] = entry.value;
          } else if (isSwitch && (SAMPLING_FIELDS.has(field) || entry.source === SOURCE_SAVED)) {
            patch[field] = entry.value;
            touchedRef.current.delete(field);
          }
        }
        write(patch);
        setLoaded(true);
      };

      return apiClient.get(`/llms/${model.id}/conversation_defaults`).then(
        (resolved) => {
          if (request === requestRef.current) apply(resolved ?? {});
        },
        (error) => {
          if (request !== requestRef.current) return;
          log.warn(
            `Could not read the conversation defaults of model ${model.id}; showing its recommended sampling`,
            error
          );
          const shown = unreadDefaults(model);
          apply(
            Object.fromEntries(
              SETTING_FIELDS.map((field) => [field, { value: shown[field], source: "model" }])
            )
          );
        }
      );
    },
    [write]
  );

  const createPayload = useCallback(() => {
    const current = valuesRef.current;
    const payload = { llm_id: current.llm_id };
    for (const field of touchedRef.current) payload[field] = current[field];
    return payload;
  }, []);

  const saveAsDefaults = useCallback(async () => {
    await flush();
    const current = valuesRef.current;
    // A value not known yet would be sent as null, which clears it: refuse.
    const unknown = ["llm_id", ...SETTING_FIELDS].filter(
      (field) => field !== "custom_prompt" && current[field] === null
    );
    if (unknown.length > 0) {
      throw new Error(`Conversation values not loaded yet: ${unknown.join(", ")}`);
    }
    return apiClient.put(`/llms/${current.llm_id}/conversation_defaults`, {
      reasoning_effort: current.reasoning_effort,
      temperature: current.temperature,
      top_p: current.top_p,
      compaction_threshold: current.compaction_threshold,
      web_search_enabled: current.web_search_enabled,
      custom_prompt: current.custom_prompt ?? "",
    });
  }, [flush]);

  const getValues = useCallback(() => valuesRef.current, []);

  return useMemo(
    () => ({
      isDraft,
      values,
      loaded,
      saveFailed,
      getValues,
      setField,
      flush,
      adopt,
      changeModel,
      selectModel,
      createPayload,
      saveAsDefaults,
    }),
    [
      isDraft,
      values,
      loaded,
      saveFailed,
      getValues,
      setField,
      flush,
      adopt,
      changeModel,
      selectModel,
      createPayload,
      saveAsDefaults,
    ]
  );
}
