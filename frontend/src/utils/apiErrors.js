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

/** `apiErrorMessage` for a body still in text form (JSON or not). */
export function apiErrorMessageFromText(status, text) {
  return apiErrorMessage(status, parseJson(text));
}

/** `apiErrorDetail` for a body still in text form; the raw text otherwise. */
export function apiErrorDetailFromText(text) {
  const parsed = parseJson(text);
  if (parsed !== null) return apiErrorDetail(parsed) ?? text;
  const trimmed = (text ?? "").trim();
  return trimmed || null;
}

function parseJson(text) {
  try {
    return JSON.parse(text ?? "");
  } catch {
    return null; // not JSON: plain text, HTML from a proxy, an empty body
  }
}
