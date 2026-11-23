// What the interface says when a backend request fails (#780).
//
// The backend writes its errors in English, with a machine code beside them:
// `{"success": false, "error": {"type": "<CODE>", "message": "..."}}` for its
// own errors (AppBaseException, src/core/exceptions.py), `{"detail": ...}` for
// FastAPI's (a validation error, an unknown route). The English sentence is
// for the log and Diagnostics; the user reads a sentence in the interface
// language, keyed on the code when the interface has one for it, else on the
// HTTP status.

import i18n from "../i18n";

/**
 * The codes a screen can show, and their sentence. Only the codes that reach a
 * page's error text are here (docs/i18n.md, "Text the backend writes"); any
 * other code says its HTTP status (`errors:api.httpStatus`).
 */
export const API_ERROR_KEYS = {
  // Every pooled database connection stayed busy (any route, 503).
  DATABASE_BUSY: "errors:api.codes.databaseBusy",
  // A database write failed (starting a download).
  DATABASE_ERROR: "errors:api.codes.database",
  // The model's folder could not be used (starting a download).
  FILESYSTEM_ERROR: "errors:api.codes.filesystem",
  // The catalog row is gone (starting a download).
  MODEL_NOT_FOUND: "errors:api.codes.modelNotFound",
  // Loading the model failed (/compact loads it).
  MODEL_LOADING_ERROR: "errors:api.codes.modelLoading",
  INSUFFICIENT_MEMORY: "errors:api.codes.insufficientMemory",
  LLM_ENGINE_FAILURE: "errors:api.codes.engineFailure",
  // An exception no handler expected (any route, 500).
  INTERNAL_SERVER_ERROR: "errors:api.codes.internal",
};

/** The backend's machine code of an error body, or null. */
export function apiErrorCode(body) {
  const code = body?.error?.type;
  return typeof code === "string" && code ? code : null;
}

/** The backend's English sentence of an error body, for the log, or null. */
export function apiErrorDetail(body) {
  const message = body?.error?.message ?? body?.detail;
  if (message === undefined || message === null || message === "") return null;
  return typeof message === "string" ? message : JSON.stringify(message);
}

/**
 * The sentence the user reads for a failed request, in the interface
 * language: the code's own when the interface has one, else the status's.
 * @param {number} status - The HTTP status of the response.
 * @param {*} body - The parsed JSON body, if any.
 * @returns {string}
 */
export function apiErrorMessage(status, body) {
  return apiErrorCodeMessage(body) ?? i18n.t("errors:api.httpStatus", { status });
}

/** The sentence of the body's code when the interface has one, else null. */
export function apiErrorCodeMessage(body) {
  const code = apiErrorCode(body);
  return code && Object.hasOwn(API_ERROR_KEYS, code) ? i18n.t(API_ERROR_KEYS[code]) : null;
}

/**
 * The error a page throws for a failed `fetch` Response: its message is the
 * interface's sentence (`apiErrorMessage`), `status` the HTTP status, `code`
 * the backend's code and `detail` the backend's English sentence, for the
 * renderer log (support reads it there).
 * @param {Response} response - A non-OK response, body not read yet.
 * @returns {Promise<Error>}
 */
export async function responseError(response) {
  let text = "";
  try {
    text = await response.text();
  } catch {
    // no readable body: the status says it
  }
  const body = parseJson(text);
  const error = new Error(apiErrorMessage(response.status, body));
  error.status = response.status;
  error.code = apiErrorCode(body);
  error.detail = body !== null ? (apiErrorDetail(body) ?? text) : text.trim() || null;
  return error;
}

/**
 * Whether `err` is a request that got no answer: `fetch` rejects with a
 * `TypeError` whose message names the fetch ("Failed to fetch", "fetch
 * failed"), the shape `client.js` retries as transient.
 */
export function isNetworkFailure(err) {
  return err instanceof TypeError && String(err.message ?? "").includes("fetch");
}

/**
 * The text a page shows for a caught request error: a request that got no
 * answer says the backend is not responding; an error without a message, or
 * any other runtime error (a `TypeError` from a bug), says the generic
 * sentence; any other error already carries a translated message (the
 * client's, `responseError`'s, or the page's own).
 */
export function requestErrorText(err) {
  if (isNetworkFailure(err)) return i18n.t("errors:connection.unreachable.title");
  if (!err?.message || err instanceof TypeError || err instanceof ReferenceError) {
    return i18n.t("errors:api.codes.internal");
  }
  return err.message;
}

/**
 * What a renderer log record carries of a caught request error: the text the
 * user read, the HTTP status and code, and the backend's English sentence
 * (`detail`), which support needs and the screen never shows.
 */
export function failureForLog(err) {
  return {
    error: String(err?.message ?? err),
    status: err?.status ?? null,
    code: typeof err?.code === "string" ? err.code : null,
    detail: err?.detail ?? null,
  };
}

function parseJson(text) {
  try {
    return JSON.parse(text ?? "");
  } catch {
    return null; // not JSON: plain text, HTML from a proxy, an empty body
  }
}
