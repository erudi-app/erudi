import { describe, it, expect } from "vitest";
import { buildBackendSpawnOptions, buildBackendEnv } from "./backendSpawn.js";

const io = { cwd: "/work/dir", env: { PYTHONUTF8: "1" } };

describe("buildBackendSpawnOptions", () => {
  it("hides the console window on every platform (#142 Windows flash)", () => {
    for (const platform of ["win32", "darwin", "linux"]) {
      expect(buildBackendSpawnOptions(platform, io).windowsHide).toBe(true);
    }
  });

  it("detaches only on POSIX (process-group kill), never on Windows (taskkill /T)", () => {
    expect(buildBackendSpawnOptions("win32", io).detached).toBe(false);
    expect(buildBackendSpawnOptions("darwin", io).detached).toBe(true);
    expect(buildBackendSpawnOptions("linux", io).detached).toBe(true);
  });

  it("pipes stdio and passes cwd/env through unchanged", () => {
    const opts = buildBackendSpawnOptions("win32", io);
    expect(opts.stdio).toEqual(["pipe", "pipe", "pipe"]);
    expect(opts.cwd).toBe("/work/dir");
    expect(opts.env).toBe(io.env);
  });
});

describe("buildBackendEnv", () => {
  it("keeps the parent environment and adds the launcher's own variables", () => {
    const env = buildBackendEnv({ PATH: "/usr/bin", HF_TOKEN: "hf_abc" });
    expect(env.PATH).toBe("/usr/bin");
    expect(env.HF_TOKEN).toBe("hf_abc");
    expect(env.PYTHONUTF8).toBe("1");
    expect(env.ERUDI_WATCH_STDIN).toBe("1");
  });

  it("drops every LangChain and LangSmith variable", () => {
    // `langsmith` ships inside the frozen backend as a transitive dependency of
    // langchain, and a single inherited variable is enough to make it POST the
    // system prompt, the knowledge-base excerpts, the question and the answer
    // to api.smith.langchain.com. A user who set one of these years ago for an
    // unrelated project must not have their conversations exfiltrated by us.
    const env = buildBackendEnv({
      PATH: "/usr/bin",
      LANGSMITH_TRACING: "true",
      LANGCHAIN_TRACING_V2: "true",
      LANGCHAIN_API_KEY: "ls_secret",
      LANGSMITH_API_KEY: "ls_secret",
      LANGCHAIN_ENDPOINT: "https://api.smith.langchain.com",
      LANGSMITH_PROJECT: "someone-elses-project",
    });
    for (const key of Object.keys(env)) {
      expect(key.startsWith("LANGCHAIN_")).toBe(false);
      expect(key.startsWith("LANGSMITH_")).toBe(false);
    }
    expect(env.PATH).toBe("/usr/bin");
  });

  it("passes the app version, so backend.log belongs to the running version (#576)", () => {
    expect(buildBackendEnv({ PATH: "/usr/bin" }, { appVersion: "1.3.0" }).ERUDI_APP_VERSION).toBe(
      "1.3.0"
    );
  });

  it("never forwards an app version inherited from the parent environment", () => {
    const env = buildBackendEnv({ ERUDI_APP_VERSION: "0.0.1" });
    expect(env.ERUDI_APP_VERSION).toBeUndefined();
  });

  it("does not mutate the environment it was given", () => {
    const parent = { LANGSMITH_TRACING: "true" };
    buildBackendEnv(parent);
    expect(parent.LANGSMITH_TRACING).toBe("true");
  });
});
