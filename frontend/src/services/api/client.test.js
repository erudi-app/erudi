// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { apiClient, tracedFetch } from "./client";

const okResponse = (body = { ok: true }) => ({
  ok: true,
  status: 200,
  json: async () => body,
});

let sendSpy;
let fetchMock;

beforeEach(() => {
  sendSpy = vi.fn();
  window.logAPI = { send: sendSpy };
  fetchMock = vi.fn().mockResolvedValue(okResponse());
  vi.stubGlobal("fetch", fetchMock);
  vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  delete window.logAPI;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const clientEntries = () =>
  sendSpy.mock.calls.map(([entry]) => entry).filter((e) => e.ns === "APIClient");
const entryData = (msg) => {
  const entry = clientEntries().find((e) => e.msg === msg);
  return entry ? JSON.parse(entry.data) : null;
};

describe("APIClient request tracing", () => {
  it("sets an X-Request-ID header on every request", async () => {
    await apiClient.get("/llms/");
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers["X-Request-ID"]).toMatch(/^fe-[0-9a-z]+-\d+$/);
  });

  it("generates a fresh request id per call", async () => {
    await apiClient.get("/a");
    await apiClient.get("/b");
    const rids = fetchMock.mock.calls.map(([, init]) => init.headers["X-Request-ID"]);
    expect(new Set(rids).size).toBe(2);
  });

  it("logs request start and completion with the same rid", async () => {
    await apiClient.post("/conversations/", { name: "hello" });

    const start = entryData("api.request");
    const done = entryData("api.response");
    expect(start).toBeTruthy();
    expect(done).toBeTruthy();
    expect(start.rid).toBe(done.rid);
    expect(start.method).toBe("POST");
    expect(start.path).toBe("/conversations/");
    expect(start.body).toContain("hello");
    expect(done.status).toBe(200);
    expect(typeof done.duration_ms).toBe("number");

    // The rid in the logs is the one that went over the wire.
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers["X-Request-ID"]).toBe(start.rid);
  });

  it("truncates the logged body preview at 500 chars", async () => {
    await apiClient.post("/conversations/", { blob: "z".repeat(600) });
    const start = entryData("api.request");
    expect(start.body.length).toBeLessThan(600);
    expect(start.body).toMatch(/… \[\+\d+\]$/);
  });

  it("logs a failure entry with rid, error, and duration on HTTP errors", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: "boom",
      json: async () => ({ detail: "kaput" }),
    });

    await expect(apiClient.get("/llms/")).rejects.toThrow("kaput");

    const fail = entryData("api.failure");
    expect(fail).toBeTruthy();
    expect(fail.error).toBe("kaput");
    expect(fail.status).toBe(500);
    expect(typeof fail.duration_ms).toBe("number");
    expect(fail.rid).toBe(entryData("api.request").rid);
  });

  it("logs a 5xx at error, naming the request", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: "boom",
      json: async () => ({ detail: "kaput" }),
    });
    await expect(apiClient.post("/llms/7/download", {})).rejects.toThrow("kaput");
    const entry = clientEntries().find((e) => e.msg === "api.failure");
    expect(entry.level).toBe("error");
    const fail = JSON.parse(entry.data);
    expect(fail.method).toBe("POST");
    expect(fail.path).toBe("/llms/7/download");
  });

  it("logs a 4xx at info: the backend refused something legitimately absent", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 404,
      statusText: "Not Found",
      json: async () => ({ detail: "Model 'ghost' not found" }),
    });
    await expect(apiClient.get("/llms/ghost")).rejects.toThrow("not found");
    const entry = clientEntries().find((e) => e.msg === "api.failure");
    expect(entry.level).toBe("info");
    expect(JSON.parse(entry.data).status).toBe(404);
    expect(clientEntries().some((e) => e.level === "error")).toBe(false);
  });
});

describe("tracedFetch", () => {
  it("sets an X-Request-ID header and merges caller headers", async () => {
    await tracedFetch("http://127.0.0.1:8765/erudi/conversations/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "hello" }),
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://127.0.0.1:8765/erudi/conversations/");
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(init.headers["X-Request-ID"]).toMatch(/^fe-[0-9a-z]+-\d+$/);
  });

  it("returns the exact Response object without touching the body", async () => {
    const body = new ReadableStream({ start() {} });
    const jsonSpy = vi.fn();
    const response = { ok: true, status: 200, body, json: jsonSpy };
    fetchMock.mockResolvedValueOnce(response);

    const result = await tracedFetch("http://127.0.0.1:8765/erudi/conversations/1/query");

    expect(result).toBe(response);
    expect(result.body).toBe(body);
    expect(result.body.locked).toBe(false); // stream untouched, still readable
    expect(jsonSpy).not.toHaveBeenCalled();
  });

  it("logs api.request and api.response with the rid that went over the wire", async () => {
    await tracedFetch("http://127.0.0.1:8765/erudi/llms/local");

    const start = entryData("api.request");
    const done = entryData("api.response");
    expect(start.method).toBe("GET");
    expect(start.path).toBe("/erudi/llms/local");
    expect(done.status).toBe(200);
    expect(typeof done.duration_ms).toBe("number");
    expect(start.rid).toBe(done.rid);

    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers["X-Request-ID"]).toBe(start.rid);
  });

  it("rethrows the raw error, logs api.failure, and never retries", async () => {
    const netError = new TypeError("Failed to fetch");
    fetchMock.mockRejectedValueOnce(netError);

    await expect(tracedFetch("http://127.0.0.1:8765/erudi/llms/local")).rejects.toBe(netError);

    expect(fetchMock).toHaveBeenCalledTimes(1); // no retry, unlike apiClient
    const fail = entryData("api.failure");
    expect(fail.error).toBe("Failed to fetch");
    expect(fail.path).toBe("/erudi/llms/local");
    expect(typeof fail.duration_ms).toBe("number");
    expect(fail.rid).toBe(entryData("api.request").rid);
  });

  it("logs a caller's own abort at info, not as a failure", async () => {
    const abort = new Error("The user aborted a request.");
    abort.name = "AbortError";
    fetchMock.mockRejectedValueOnce(abort);

    await expect(tracedFetch("http://127.0.0.1:8765/erudi/arena/1/query")).rejects.toBe(abort);
    const entry = clientEntries().find((e) => e.msg === "api.failure");
    expect(entry.level).toBe("info");
  });

  it("logs a FormData body as kind and size, not a preview", async () => {
    const form = new FormData();
    form.append("file", "a");
    form.append("name", "b");

    await tracedFetch("http://127.0.0.1:8765/erudi/knowledge_base/create", {
      method: "POST",
      body: form,
    });

    const start = entryData("api.request");
    expect(start.body).toBeUndefined();
    expect(start.body_kind).toBe("FormData");
    expect(start.body_size).toBe(2);

    // The body itself is passed through untouched.
    const [, init] = fetchMock.mock.calls[0];
    expect(init.body).toBe(form);
  });

  it("truncates the logged string body preview at 500 chars", async () => {
    await tracedFetch("http://127.0.0.1:8765/erudi/conversations/", {
      method: "POST",
      body: JSON.stringify({ blob: "z".repeat(600) }),
    });

    const start = entryData("api.request");
    expect(start.body.length).toBeLessThan(600);
    expect(start.body).toMatch(/… \[\+\d+\]$/);
  });
});
