import { createLogger } from "../../utils/logger";
import { getApiBaseUrl } from "../../config/api";
import { truncateValue } from "../../utils/interactionLogger";
import { reportNetworkFailure, reportNetworkSuccess } from "../../utils/networkStatus";
import i18n from "../../i18n";

const log = createLogger("APIClient");

// Per-request correlation id, sent as X-Request-ID. The backend echoes it and
// injects it in its own logs, so the same id on both sides stitches an
// end-to-end trace. A retried request gets a fresh id per attempt (each HTTP
// exchange is individually traceable).
let requestCounter = 0;
const nextRequestId = () => {
  requestCounter += 1;
  return `fe-${Date.now().toString(36)}-${requestCounter}`;
};

const BODY_PREVIEW_MAX_CHARS = 500;

/** Path + query of an absolute URL for log entries; falls back to the raw string. */
function pathForLog(url) {
  try {
    const parsed = new URL(url);
    return `${parsed.pathname}${parsed.search}`;
  } catch {
    return String(url);
  }
}

/**
 * Whether a rejected request died at the network layer rather than answering.
 *
 * The status pill reads connectivity from the operating system, which reports a
 * link, not a reachable internet. A request that came back with a status —
 * however bad — proves the path works; one that came back with nothing is the
 * evidence that it does not. A caller-side abort (our own timeout, an unmounting
 * component) says nothing about the network and is excluded.
 * @param {Error} error - The rejection from a request attempt.
 * @returns {boolean}
 */
function isNetworkLevelError(error) {
  return error?.name !== "AbortError" && error?.status === undefined;
}

/**
 * Loggable preview of a request body. Strings are truncated; FormData and
 * binary payloads are summarized as kind + size instead of being dumped.
 * @param {*} body - The fetch options.body, if any.
 * @returns {Object} Fields to spread into the api.request log entry.
 */
function describeBody(body) {
  if (body === undefined || body === null) {
    return {};
  }
  if (typeof body === "string") {
    return { body: truncateValue(body, BODY_PREVIEW_MAX_CHARS) };
  }
  if (typeof URLSearchParams !== "undefined" && body instanceof URLSearchParams) {
    return { body: truncateValue(body.toString(), BODY_PREVIEW_MAX_CHARS) };
  }
  if (typeof FormData !== "undefined" && body instanceof FormData) {
    return { body_kind: "FormData", body_size: [...body.keys()].length };
  }
  if (typeof Blob !== "undefined" && body instanceof Blob) {
    return { body_kind: "Blob", body_size: body.size };
  }
  if (body instanceof ArrayBuffer) {
    return { body_kind: "ArrayBuffer", body_size: body.byteLength };
  }
  if (ArrayBuffer.isView(body)) {
    return { body_kind: body.constructor.name, body_size: body.byteLength };
  }
  return { body_kind: typeof body };
}

/**
 * Drop-in replacement for the global `fetch` that adds request tracing.
 *
 * Behaves exactly like `fetch`: no retry, no timeout, no JSON parsing, no
 * ok-check — the raw `Response` is returned (or the raw error rethrown) and
 * the body is never touched, so streaming readers keep working. The only
 * additions are the `X-Request-ID` header (merged into caller headers) and
 * api.request / api.response / api.failure log entries, giving raw call sites
 * the same click→request correlation as `apiClient`. api.response is logged
 * when the headers arrive, before the body is consumed.
 *
 * @param {string} url - Absolute request URL (call sites build it from API_BASE_URL)
 * @param {Object} [options] - Standard fetch options, passed through untouched
 * @returns {Promise<Response>} The raw fetch Response
 */
export async function tracedFetch(url, options = {}) {
  const rid = nextRequestId();
  const method = options.method || "GET";
  const startedAt = Date.now();

  log.info("api.request", {
    rid,
    method,
    path: pathForLog(url),
    ...describeBody(options.body),
  });

  const callerHeaders =
    typeof Headers !== "undefined" && options.headers instanceof Headers
      ? Object.fromEntries(options.headers.entries())
      : options.headers;

  try {
    const response = await fetch(url, {
      ...options,
      headers: { ...callerHeaders, "X-Request-ID": rid },
    });
    reportNetworkSuccess();
    log.info("api.response", {
      rid,
      status: response.status,
      duration_ms: Date.now() - startedAt,
    });
    return response;
  } catch (error) {
    if (isNetworkLevelError(error)) reportNetworkFailure();
    // A caller that aborted its own request (the arena's Stop button) got
    // what it asked for: that is an INFO, not a failure.
    const write = error?.name === "AbortError" ? log.info : log.error;
    write("api.failure", {
      rid,
      method: options.method || "GET",
      path: pathForLog(url),
      error: error.message,
      duration_ms: Date.now() - startedAt,
    });
    throw error;
  }
}

/**
 * The level of a request that ended with `status`, per docs/logging.md: a
 * 5xx (or no status at all: network, timeout) is a failure the user needed
 * and did not get; a 4xx is the backend refusing something that legitimately
 * is not there or not allowed, which the backend logs itself at INFO and
 * which must not fill the Diagnostics panel from this side either.
 * @param {number|undefined} status - HTTP status, or undefined when none came back.
 * @returns {"error"|"info"} The logger method to use.
 */
function failureLevel(status) {
  return typeof status === "number" && status < 500 ? "info" : "error";
}

/**
 * API client with built-in retry logic, timeout handling, and error normalization
 * Provides consistent error handling and response transformation across the app
 */
class APIClient {
  constructor(baseURL = null) {
    // null → resolve the live base URL per request (follows the dynamic backend
    // port). Pass an explicit baseURL only to pin the client to a fixed host.
    this.baseURL = baseURL;
    this.timeout = 30000; // 30 seconds
    this.maxRetries = 3;
    this.retryDelay = 1000; // 1 second, will exponentially backoff
  }

  /**
   * Make a GET request
   * @param {string} endpoint - Relative endpoint path
   * @param {Object} options - Request options (headers, params, etc)
   * @returns {Promise<*>} Parsed response data
   */
  async get(endpoint, options = {}) {
    return this.request(endpoint, { method: "GET", ...options });
  }

  /**
   * Make a POST request
   * @param {string} endpoint - Relative endpoint path
   * @param {Object} data - Request body data
   * @param {Object} options - Request options
   * @returns {Promise<*>} Parsed response data
   */
  async post(endpoint, data = {}, options = {}) {
    return this.request(endpoint, {
      method: "POST",
      body: JSON.stringify(data),
      ...options,
    });
  }

  /**
   * Make a PUT request
   * @param {string} endpoint - Relative endpoint path
   * @param {Object} data - Request body data
   * @param {Object} options - Request options
   * @returns {Promise<*>} Parsed response data
   */
  async put(endpoint, data = {}, options = {}) {
    return this.request(endpoint, {
      method: "PUT",
      body: JSON.stringify(data),
      ...options,
    });
  }

  /**
   * Make a PATCH request
   * @param {string} endpoint - Relative endpoint path
   * @param {Object} data - Request body data
   * @param {Object} options - Request options
   * @returns {Promise<*>} Parsed response data
   */
  async patch(endpoint, data = {}, options = {}) {
    return this.request(endpoint, {
      method: "PATCH",
      body: JSON.stringify(data),
      ...options,
    });
  }

  /**
   * Make a DELETE request
   * @param {string} endpoint - Relative endpoint path
   * @param {Object} options - Request options
   * @returns {Promise<*>} Parsed response data
   */
  async delete(endpoint, options = {}) {
    return this.request(endpoint, { method: "DELETE", ...options });
  }

  /**
   * Core request method with retry logic and error handling
   * @private
   * @param {string} endpoint - API endpoint
   * @param {Object} options - Fetch options, plus one client-only flag:
   *   `silentFailure` (boolean) logs a failed request's `api.failure` entry
   *   at `info` instead of the usual level. For a caller that OBSERVES its
   *   own health by polling this endpoint (the bug-icon counter's poll of
   *   `/diagnostics/`, see `shared/hooks/useBugCounter.js`) — a backend that
   *   is down is already surfaced by `ConnectionStatus` and the error
   *   screen, and logging that failure at `error` would feed the very
   *   Diagnostics list/badge the poll exists to read, growing forever on
   *   every tick. Never set this for a request a person asked for: it
   *   should still count once, like any other real failure. Not forwarded
   *   to `fetch()`.
   * @param {number} attempt - Current attempt number
   * @returns {Promise<*>} Parsed response
   */
  async request(endpoint, options = {}, attempt = 1) {
    const { silentFailure = false, ...fetchOptions } = options;
    const url = `${this.baseURL || getApiBaseUrl()}${endpoint}`;
    const controller = new AbortController();
    const rid = nextRequestId();
    const method = options.method || "GET";
    const startedAt = Date.now();

    log.info("api.request", {
      rid,
      method,
      path: endpoint,
      attempt,
      body:
        typeof options.body === "string"
          ? truncateValue(options.body, BODY_PREVIEW_MAX_CHARS)
          : undefined,
    });

    try {
      // Set timeout
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const headers = {
        "Content-Type": "application/json",
        ...fetchOptions.headers,
        "X-Request-ID": rid,
      };

      const response = await fetch(url, {
        ...fetchOptions,
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      // A status came back, so the network path works whatever the status says.
      reportNetworkSuccess();

      // Handle non-OK responses
      if (!response.ok) {
        const error = await this.handleErrorResponse(response);
        throw error;
      }

      log.info("api.response", {
        rid,
        status: response.status,
        duration_ms: Date.now() - startedAt,
      });

      // Parse and return response
      try {
        const data = await response.json();
        return data;
      } catch {
        // If response is not JSON, return response object
        return response;
      }
    } catch (error) {
      const durationMs = Date.now() - startedAt;

      // Handle timeout. The thrown message is user-facing (pages surface
      // `err.message`), so it is translated; the log entry stays English.
      if (error.name === "AbortError") {
        const timeoutError = new Error(i18n.t("errors:api.timeout"));
        timeoutError.code = "TIMEOUT";
        (silentFailure ? log.info : log.error)("api.failure", {
          rid,
          method,
          path: endpoint,
          error: "Request timeout",
          duration_ms: durationMs,
        });
        throw timeoutError;
      }

      // Retry on transient errors
      if (this.isTransientError(error) && attempt < this.maxRetries) {
        const delay = this.retryDelay * Math.pow(2, attempt - 1); // Exponential backoff
        log.warn(`Request failed, retrying in ${delay}ms (attempt ${attempt}/${this.maxRetries})`, {
          rid,
          error: error.message,
        });
        await this.sleep(delay);
        return this.request(endpoint, options, attempt + 1);
      }

      // Only once the request is really over: a retry that succeeds says the
      // network is fine, and reporting each attempt would flicker the pill.
      if (isNetworkLevelError(error)) reportNetworkFailure();

      const level = silentFailure ? "info" : failureLevel(error.status);
      log[level]("api.failure", {
        rid,
        method,
        path: endpoint,
        error: error.message,
        status: error.status,
        duration_ms: durationMs,
      });
      throw error;
    }
  }

  /**
   * Handle error responses from API
   * @private
   * @param {Response} response - Fetch Response object
   * @returns {Promise<Error>} Normalized error
   */
  async handleErrorResponse(response) {
    let errorData = {};
    try {
      errorData = await response.json();
    } catch {
      // If response is not JSON, use status text
      errorData = { detail: response.statusText };
    }

    // The backend's own `detail`/`message` is passed through as-is; only the
    // generic fallback is translated (it is what pages show the user).
    const error = new Error(
      errorData.detail ||
        errorData.message ||
        i18n.t("errors:api.httpStatus", { status: response.status })
    );
    error.status = response.status;
    error.code = `HTTP_${response.status}`;
    error.data = errorData;

    return error;
  }

  /**
   * Check if error is transient and should be retried
   * @private
   * @param {Error} error - Error object
   * @returns {boolean} Whether error is transient
   */
  isTransientError(error) {
    // Network errors are transient
    if (error instanceof TypeError && error.message.includes("fetch")) {
      return true;
    }

    // Specific error codes that are transient
    const transientCodes = ["ECONNREFUSED", "ENOTFOUND", "ETIMEDOUT", "EHOSTUNREACH"];
    if (transientCodes.includes(error.code)) {
      return true;
    }

    return false;
  }

  /**
   * Sleep utility for delays
   * @private
   * @param {number} ms - Milliseconds to sleep
   * @returns {Promise<void>}
   */
  sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// Export singleton instance
export const apiClient = new APIClient();

export default apiClient;
