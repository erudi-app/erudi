import { describe, it, expect, beforeAll, afterAll } from "vitest";
import i18n from "../i18n";
import fr from "../locales/fr/errors.json";
import {
  API_ERROR_KEYS,
  apiErrorCode,
  apiErrorDetailFromText,
  apiErrorMessage,
  apiErrorMessageFromText,
} from "./apiErrors";

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

  it("reads a body still in text form, JSON or not", () => {
    expect(apiErrorMessageFromText(503, JSON.stringify(appError("DATABASE_BUSY", "busy")))).toBe(
      fr.api.codes.databaseBusy
    );
    expect(apiErrorMessageFromText(502, "<html>Bad Gateway</html>")).toBe("Erreur API : 502");
  });
});

describe("apiErrorCode", () => {
  it("is the backend's error type, or null", () => {
    expect(apiErrorCode(appError("DATABASE_BUSY", "x"))).toBe("DATABASE_BUSY");
    expect(apiErrorCode({ detail: "x" })).toBeNull();
    expect(apiErrorCode(undefined)).toBeNull();
  });
});

describe("apiErrorDetailFromText: the backend's own sentence, for the log", () => {
  it("unwraps both of the backend's envelopes", () => {
    expect(apiErrorDetailFromText('{"detail":"Model 14 not found"}')).toBe("Model 14 not found");
    expect(apiErrorDetailFromText(JSON.stringify(appError("MODEL_NOT_FOUND", "gone")))).toBe(
      "gone"
    );
  });

  it("keeps a structured detail as JSON and any other text as it is", () => {
    expect(apiErrorDetailFromText('{"detail":[{"msg":"Field required"}]}')).toBe(
      '[{"msg":"Field required"}]'
    );
    expect(apiErrorDetailFromText('{"error":"x"}')).toBe('{"error":"x"}');
    expect(apiErrorDetailFromText("  Bad Gateway \n")).toBe("Bad Gateway");
    expect(apiErrorDetailFromText("")).toBeNull();
    expect(apiErrorDetailFromText(undefined)).toBeNull();
  });
});
