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
// How long the "could not be saved" note stays when nothing clears it sooner.
export const SAVE_FAILED_NOTE_MS = 6000;
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
// recommended sampling and the shipped defaults. Display only, and marked
// unread: a value the user did not set is still left to the backend at
// creation, and /config global refuses to save it.
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

const notLoaded = (fields) => new Error(`Conversation values not loaded yet: ${fields.join(", ")}`);

// Per-conversation bookkeeping: the last value the server holds per field,
// the latest save per field, the saves on their way per field, and a count
// of the changes made (saves and model switches) a load is stamped with.
const newBook = () => ({ saved: {}, latest: {}, inFlight: {}, changes: 0 });

/**
 * The one owner of a conversation's settings (#609): the header panel,
 * /config, /config global, /effort and /model all read and write through it,
 * so two views can never disagree.
 *
 * In a conversation (`conversationId` a real id) every change is saved with a
 * one-field PATCH: effort, web search, threshold and instructions at once,
 * creativity and diversity 300 ms after the slider stops (they also ride in
 * each turn's payload). Saves go out one after another, in the order they
 * were made, each tied to the conversation it was made in: a save of A that
 * lands after the page moved to B never touches B, and B's saves and sends
 * never wait for A's (one save chain per conversation). A failed save re-reads
 * the conversation and shows what the server holds (the last saved value
 * when that read fails too), unless a later change of the same control is on
 * its way, and raises `saveFailed`, cleared by the next successful save, a
 * conversation switch or after a few seconds. `flush()` sends what is pending
 * and resolves once every save of the current conversation has settled: the
 * send path and /model wait for it. A load takes `loadStamp()` when it starts and passes it to
 * `adopt(conversation, stamp)`, which drops a row of a conversation the page
 * has left, or older than a change made meanwhile. `changeModel(llmId)` sends only `{llm_id}` and adopts the
 * conversation the backend returns, with the new model's values applied
 * server-side.
 *
 * On the new-conversation page (`DRAFT`) nothing is saved: `selectModel(model)`
 * reads that model's resolved defaults (`GET /llms/{id}/conversation_defaults`),
 * a late answer for a previous model is ignored, and the values the user set
 * are remembered. On a model switch, the values the user did not set take the
 * new model's; those set before the switch follow the /model rule (the new
 * model's creativity and diversity replace them at once, its saved values
 * when they arrive, the others stay); a value set after the switch started
 * stays. `ready()` resolves once the current model's defaults are in, and
 * `createPayload()` is then the creation body: the model and the values the
 * user set, every other one resolved by the backend.
 *
 * `saveAsDefaults()` (/config global) saves the six current values for the
 * current model; empty instructions clear the instructions saved for it. It
 * refuses while a value is not known: on the draft, while the model's
 * defaults are on their way, and while values shown from a failed read are
 * not read again (it retries the read first).
 */
export default function useConversationSettings(conversationId) {
  const isDraft = conversationId === DRAFT;
  const [values, setValues] = useState(EMPTY);
  const [loaded, setLoaded] = useState(false);
  const [saveFailed, setSaveFailed] = useState(false);

  // Synchronous mirrors: the send path reads the values right after adopt().
  const valuesRef = useRef(EMPTY);
  const loadedRef = useRef(false);
  const idRef = useRef(conversationId);
  idRef.current = conversationId;

  // Conversation: the bookkeeping per conversation id, the save sequence,
  // the save chain per conversation id, the pending slider save and its timer.
  const booksRef = useRef(new Map());
  const saveSeqRef = useRef(0);
  const chainsRef = useRef(new Map());
  const pendingSamplingRef = useRef({ id: null, patch: {} });
  const timerRef = useRef(null);
  const failedTimerRef = useRef(null);

  // Draft: when each field was set (a touch sequence), the fields shown from
  // a failed read, the current model and its defaults request.
  const touchedRef = useRef(new Map());
  const touchSeqRef = useRef(0);
  const unreadRef = useRef(new Set());
  const modelRef = useRef(null);
  const requestRef = useRef(0);
  const pendingRequestRef = useRef(Promise.resolve());

  const write = useCallback((patch) => {
    valuesRef.current = { ...valuesRef.current, ...patch };
    setValues(valuesRef.current);
  }, []);

  const markLoaded = useCallback((value) => {
    loadedRef.current = value;
    setLoaded(value);
  }, []);

  const book = useCallback((id) => {
    let entry = booksRef.current.get(id);
    if (!entry) {
      entry = newBook();
      booksRef.current.set(id, entry);
    }
    return entry;
  }, []);

  const busy = useCallback(
    (id, field) =>
      (book(id).inFlight[field] ?? 0) > 0 ||
      (pendingSamplingRef.current.id === id &&
        Object.hasOwn(pendingSamplingRef.current.patch, field)),
    [book]
  );

  const clearFailure = useCallback(() => {
    clearTimeout(failedTimerRef.current);
    setSaveFailed(false);
  }, []);

  // One save chain per conversation: a save of A, slow or failing, never
  // delays a save or a send in B.
  const chain = useCallback((id) => chainsRef.current.get(id) ?? Promise.resolve(), []);
  const enqueue = useCallback(
    (id, task) => {
      const next = chain(id).then(task);
      chainsRef.current.set(id, next);
      return next;
    },
    [chain]
  );

  const reportFailure = useCallback(() => {
    setSaveFailed(true);
    clearTimeout(failedTimerRef.current);
    failedTimerRef.current = setTimeout(() => setSaveFailed(false), SAVE_FAILED_NOTE_MS);
  }, []);

  // What the server holds now, after a save or a switch that failed (it may
  // have been applied all the same: a timeout, a retried request).
  const reread = useCallback(async (id) => {
    try {
      const row = await apiClient.get(`/conversations/${id}`);
      return row ? fromConversation(row) : null;
    } catch (error) {
      log.warn(`Could not re-read conversation ${id} after a failed save`, error);
      return null;
    }
  }, []);

  // A conversation row's values into the state and the bookkeeping, except
  // a value the user is still saving.
  const takeRow = useCallback(
    (id, row) => {
      const entry = book(id);
      const patch = {};
      for (const [field, value] of Object.entries(row)) {
        entry.saved[field] = value;
        if (!busy(id, field)) patch[field] = value;
      }
      write(patch);
      markLoaded(true);
    },
    [book, busy, write, markLoaded]
  );

  const enqueuePatch = useCallback(
    (id, patch) => {
      const entry = book(id);
      entry.changes += 1;
      const seq = ++saveSeqRef.current;
      const fields = Object.keys(patch);
      for (const field of fields) {
        entry.latest[field] = seq;
        entry.inFlight[field] = (entry.inFlight[field] ?? 0) + 1;
      }
      const run = async () => {
        try {
          await apiClient.patch(`/conversations/${id}`, patch);
          for (const field of fields) entry.saved[field] = patch[field];
          if (id === idRef.current) clearFailure();
        } catch (error) {
          log.error(`Failed to save ${fields.join(", ")} of conversation ${id}`, error);
          const fresh = await reread(id);
          const back = {};
          for (const field of fields) {
            if (fresh) entry.saved[field] = fresh[field];
            const superseded =
              entry.latest[field] !== seq ||
              (pendingSamplingRef.current.id === id &&
                Object.hasOwn(pendingSamplingRef.current.patch, field));
            if (!superseded) back[field] = entry.saved[field];
          }
          if (id === idRef.current) {
            write(back);
            reportFailure();
          }
        } finally {
          for (const field of fields) entry.inFlight[field] -= 1;
        }
      };
      return enqueue(id, run);
    },
    [book, enqueue, write, reread, clearFailure, reportFailure]
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
    markLoaded(false);
    clearFailure();
    touchedRef.current = new Map();
    unreadRef.current = new Set();
    modelRef.current = null;
    requestRef.current += 1;
    pendingRequestRef.current = Promise.resolve();
    return () => sendPendingSampling();
  }, [conversationId, sendPendingSampling, markLoaded, clearFailure]);

  useEffect(() => () => clearTimeout(failedTimerRef.current), []);

  const flush = useCallback(() => {
    sendPendingSampling();
    return chain(idRef.current);
  }, [sendPendingSampling, chain]);

  const setField = useCallback(
    (field, value) => {
      write({ [field]: value });
      if (idRef.current === DRAFT) {
        touchedRef.current.set(field, ++touchSeqRef.current);
        unreadRef.current.delete(field);
        return;
      }
      if (SAMPLING_FIELDS.has(field)) {
        const pending = pendingSamplingRef.current;
        pendingSamplingRef.current = {
          id: idRef.current,
          patch: { ...(pending.id === idRef.current ? pending.patch : {}), [field]: value },
        };
        clearTimeout(timerRef.current);
        timerRef.current = setTimeout(sendPendingSampling, SAMPLING_SAVE_DELAY_MS);
        return;
      }
      enqueuePatch(idRef.current, { [field]: value });
    },
    [write, enqueuePatch, sendPendingSampling]
  );

  const loadStamp = useCallback(
    () => ({ id: idRef.current, changes: book(idRef.current).changes }),
    [book]
  );

  // A conversation row from a load: dropped when the page has moved to
  // another conversation since the load started, or when a save or a model
  // switch was made meanwhile (`stamp`, from loadStamp()); its values are then
  // another conversation's, or older than what the page shows.
  const adopt = useCallback(
    (conversation, stamp) => {
      const id = idRef.current;
      if (stamp && (stamp.id !== id || stamp.changes !== book(id).changes)) return false;
      takeRow(id, fromConversation(conversation));
      return true;
    },
    [book, takeRow]
  );

  const changeModel = useCallback(
    async (llmId) => {
      sendPendingSampling();
      const id = idRef.current;
      const entry = book(id);
      entry.changes += 1;
      entry.inFlight.llm_id = (entry.inFlight.llm_id ?? 0) + 1;
      let switched = null;
      await enqueue(id, async () => {
        let row = null;
        let failed = false;
        try {
          switched = await apiClient.patch(`/conversations/${id}`, { llm_id: llmId });
          row = fromConversation(switched);
        } catch (error) {
          log.error(`Failed to switch conversation ${id} to model ${llmId}`, error);
          failed = true;
          row = await reread(id);
        } finally {
          entry.inFlight.llm_id -= 1;
        }
        if (id !== idRef.current) return;
        if (row) takeRow(id, row);
        if (failed) reportFailure();
        else clearFailure();
      });
      return switched;
    },
    [book, enqueue, takeRow, reread, sendPendingSampling, reportFailure, clearFailure]
  );

  // The model's defaults into the draft. `switchSeq` is the touch sequence
  // when a model switch started (null on a first pick or a re-read): a field
  // set before it follows the /model rule, one set after it stays.
  const fetchDefaults = useCallback(
    (model, switchSeq) => {
      const request = ++requestRef.current;
      markLoaded(false);
      const apply = (incoming, unread) => {
        const patch = {};
        for (const field of SETTING_FIELDS) {
          const entry = incoming[field];
          if (!entry) continue;
          const touchedAt = touchedRef.current.get(field);
          if (touchedAt === undefined) {
            patch[field] = entry.value;
            if (unread) unreadRef.current.add(field);
            else unreadRef.current.delete(field);
          } else if (
            switchSeq !== null &&
            touchedAt <= switchSeq &&
            entry.source === SOURCE_SAVED
          ) {
            patch[field] = entry.value;
            touchedRef.current.delete(field);
            unreadRef.current.delete(field);
          }
        }
        write(patch);
        markLoaded(true);
      };
      const pending = apiClient.get(`/llms/${model.id}/conversation_defaults`).then(
        (resolved) => {
          if (request === requestRef.current) apply(resolved ?? {}, false);
        },
        (error) => {
          if (request !== requestRef.current) return;
          log.warn(
            `Could not read the conversation defaults of model ${model.id}; showing its recommended sampling`,
            error
          );
          const shown = unreadDefaults(model);
          apply(
            Object.fromEntries(SETTING_FIELDS.map((field) => [field, { value: shown[field] }])),
            true
          );
        }
      );
      pendingRequestRef.current = pending;
      return pending;
    },
    [write, markLoaded]
  );

  const selectModel = useCallback(
    (model) => {
      const previous = valuesRef.current.llm_id;
      if (!model || model.id === previous) return Promise.resolve();
      const isSwitch = previous !== null;
      modelRef.current = model;
      write({ llm_id: model.id });
      if (isSwitch) {
        // The new model's creativity and diversity replace the ones set for
        // the previous model, whatever its saved values: never sent for it.
        for (const field of SAMPLING_FIELDS) touchedRef.current.delete(field);
      }
      return fetchDefaults(model, isSwitch ? touchSeqRef.current : null);
    },
    [write, fetchDefaults]
  );

  // Resolves once the defaults of the current model are in (a later pick
  // started meanwhile is waited for too).
  const ready = useCallback(async () => {
    let pending;
    do {
      pending = pendingRequestRef.current;
      await pending;
    } while (pending !== pendingRequestRef.current);
  }, []);

  const createPayload = useCallback(() => {
    const current = valuesRef.current;
    const payload = { llm_id: current.llm_id };
    for (const field of touchedRef.current.keys()) payload[field] = current[field];
    return payload;
  }, []);

  const saveAsDefaults = useCallback(async () => {
    await flush();
    if (idRef.current === DRAFT) {
      if (!loadedRef.current) throw notLoaded(["defaults"]);
      if (unreadRef.current.size > 0 && modelRef.current) {
        // Shown from a failed read: read them again rather than save them.
        await fetchDefaults(modelRef.current, null);
        if (unreadRef.current.size > 0) throw notLoaded([...unreadRef.current]);
      }
    }
    const current = valuesRef.current;
    // A value not known yet would be sent as null, which clears it: refuse.
    const unknown = ["llm_id", ...SETTING_FIELDS].filter(
      (field) => field !== "custom_prompt" && current[field] === null
    );
    if (unknown.length > 0) throw notLoaded(unknown);
    return apiClient.put(`/llms/${current.llm_id}/conversation_defaults`, {
      reasoning_effort: current.reasoning_effort,
      temperature: current.temperature,
      top_p: current.top_p,
      compaction_threshold: current.compaction_threshold,
      web_search_enabled: current.web_search_enabled,
      custom_prompt: current.custom_prompt ?? "",
    });
  }, [flush, fetchDefaults]);

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
      loadStamp,
      adopt,
      changeModel,
      selectModel,
      ready,
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
      loadStamp,
      adopt,
      changeModel,
      selectModel,
      ready,
      createPayload,
      saveAsDefaults,
    ]
  );
}
