import { describe, it, expect, vi } from "vitest";
import { singleFlight } from "./singleFlight";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("singleFlight", () => {
  it("hands a call made while one is running the same promise, without a second run", async () => {
    const run = deferred();
    const fn = vi.fn(() => run.promise);
    const once = singleFlight(fn);

    const first = once();
    const second = once();

    expect(fn).toHaveBeenCalledTimes(1);
    expect(second).toBe(first);
    run.resolve({ ok: true });
    await expect(first).resolves.toEqual({ ok: true });
  });

  it("runs again once the previous run has settled", async () => {
    const fn = vi.fn(async () => "done");
    const once = singleFlight(fn);

    await once();
    await once();

    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("propagates a rejection and lets the next call run", async () => {
    const fn = vi.fn().mockRejectedValueOnce(new Error("refused")).mockResolvedValueOnce("ok");
    const once = singleFlight(fn);

    await expect(once()).rejects.toThrow("refused");
    await expect(once()).resolves.toBe("ok");
    expect(fn).toHaveBeenCalledTimes(2);
  });
});
