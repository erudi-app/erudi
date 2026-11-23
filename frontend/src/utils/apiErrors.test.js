import { describe, it, expect, beforeAll, afterAll } from "vitest";
import i18n from "../i18n";
import fr from "../locales/fr/errors.json";
import {
  API_ERROR_KEYS,
  apiErrorCode,
  apiErrorMessage,
  failureForLog,
  requestErrorText,
  responseError,
} from "./apiErrors";

const detailOf = async (text) =>
  (await responseError({ status: 500, text: async () => text })).detail;

const appError = (type, message) => ({ success: false, error: { type, message } });

describe("apiErrorMessage (#780)", () => {
  beforeAll(async () => {
    await i18n.changeLanguage("fr");
  });
  afterAll(async () => {
    await i18n.changeLanguage("en");
  });

  it.each(Object.keys(API_ERROR_KEYS))("says %s in the interface language", (code) => {
    const key = API_ERROR_KEYS[code].replace(/^errors:/, "").split(".");
    const expected = key.reduce((node, part) => node?.[part], fr);
    expect(typeof expected).toBe("string");
    expect(apiErrorMessage(500, appError(code, "an English sentence"))).toBe(expected);
  });

  it("says the HTTP status for a code it has no sentence for", () => {
    expect(apiErrorMessage(409, appError("STATE_CONFLICT", "Cannot delete"))).toBe(
      "Erreur API : 409"
    );
  });

  it("says the HTTP status for FastAPI's detail, a missing body or an inherited name", () => {
    expect(apiErrorMessage(500, { detail: "CUDA driver version is insufficient" })).toBe(
      "Erreur API : 500"
    );
    expect(apiErrorMessage(502, null)).toBe("Erreur API : 502");
    expect(apiErrorMessage(500, appError("constructor", "x"))).toBe("Erreur API : 500");
  });
});

describe("responseError and requestErrorText", () => {
  it("reads a failed response's code, or its status when the body is not JSON", async () => {
    const busy = {
      status: 503,
      text: async () => JSON.stringify(appError("DATABASE_BUSY", "Every connection was in use")),
    };
    const error = await responseError(busy);
    expect(error.message).toBe(i18n.t("errors:api.codes.databaseBusy"));
    expect(error.status).toBe(503);
    expect(error.code).toBe("DATABASE_BUSY");
    // The English sentence, for the renderer log.
    expect(error.detail).toBe("Every connection was in use");

    const html = { status: 502, text: async () => "<html>Bad Gateway</html>" };
    expect((await responseError(html)).message).toBe(
      i18n.t("errors:api.httpStatus", { status: 502 })
    );
    expect((await responseError(html)).detail).toBe("<html>Bad Gateway</html>");
  });

  it("says a request that got no answer in the interface language, not the runtime's text", () => {
    expect(requestErrorText(new TypeError("Failed to fetch"))).toBe(
      i18n.t("errors:connection.unreachable.title")
    );
    expect(requestErrorText(new Error("Erreur API : 500"))).toBe("Erreur API : 500");
  });

  it("says the generic sentence for a code TypeError, never the backend-down one", () => {
    const bug = new TypeError("Cannot read properties of undefined (reading 'name')");
    expect(requestErrorText(bug)).toBe(i18n.t("errors:api.codes.internal"));
    expect(requestErrorText(new Error(""))).toBe(i18n.t("errors:api.codes.internal"));
  });
});

describe("apiErrorCode", () => {
  it("is the backend's error type, or null", () => {
    expect(apiErrorCode(appError("DATABASE_BUSY", "x"))).toBe("DATABASE_BUSY");
    expect(apiErrorCode({ detail: "x" })).toBeNull();
    expect(apiErrorCode(undefined)).toBeNull();
  });
});

describe("failureForLog", () => {
  it("keeps the backend's English sentence beside the text the user read", async () => {
    const error = await responseError({
      status: 503,
      text: async () => JSON.stringify(appError("DATABASE_BUSY", "Every connection was in use")),
    });
    expect(failureForLog(error)).toEqual({
      error: error.message,
      status: 503,
      code: "DATABASE_BUSY",
      detail: "Every connection was in use",
    });
    expect(failureForLog(new TypeError("Failed to fetch"))).toMatchObject({
      error: "Failed to fetch",
      detail: null,
    });
  });
});

describe("responseError's detail: the backend's own sentence, for the log", () => {
  it("unwraps both of the backend's envelopes", async () => {
    expect(await detailOf('{"detail":"Model 14 not found"}')).toBe("Model 14 not found");
    expect(await detailOf(JSON.stringify(appError("MODEL_NOT_FOUND", "gone")))).toBe("gone");
  });

  it("keeps a structured detail as JSON and any other text as it is", async () => {
    expect(await detailOf('{"detail":[{"msg":"Field required"}]}')).toBe(
      '[{"msg":"Field required"}]'
    );
    expect(await detailOf('{"error":"x"}')).toBe('{"error":"x"}');
    expect(await detailOf("  Bad Gateway \n")).toBe("Bad Gateway");
    expect(await detailOf("")).toBeNull();
  });
});
