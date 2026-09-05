import { describe, it, expect } from "vitest";
import { describeEngineFailure, isEngineNotice } from "./engineNotice";

describe("isEngineNotice", () => {
  it("recognises the startup event", () => {
    expect(isEngineNotice({ event: "engine_notice", code: "CUDA_ERROR" })).toBe(true);
  });

  it("ignores every other lifecycle event", () => {
    expect(isEngineNotice({ event: "ready", port: 27182 })).toBe(false);
    expect(isEngineNotice({ event: "phase", phase: "loading_catalog" })).toBe(false);
    expect(isEngineNotice(null)).toBe(false);
    expect(isEngineNotice(undefined)).toBe(false);
  });
});

describe("describeEngineFailure", () => {
  it("describes a card below the compute-capability floor", () => {
    const described = describeEngineFailure({
      event: "engine_notice",
      code: "CUDA_COMPUTE_CAPABILITY_TOO_LOW",
      gpu_name: "NVIDIA GeForce GTX 780",
      compute_capability: "3.5",
      driver_cuda_version: "12.8",
      required_cuda_version: "12.8",
      raw: "compute capability 3.5 is below 5.0",
    });

    expect(described.code).toBe("CUDA_COMPUTE_CAPABILITY_TOO_LOW");
    expect(described.gpuName).toBe("NVIDIA GeForce GTX 780");
    expect(described.computeCapability).toBe("3.5");
    expect(described.raw).toBe("compute capability 3.5 is below 5.0");
    // The copy names the card and its capability, and says a driver won't help.
    expect(described.detail).toContain("NVIDIA GeForce GTX 780");
    expect(described.detail).toContain("3.5");
    expect(described.title).toBeTruthy();
    expect(described.hint).toBeTruthy();
  });

  it("names the required driver version for a driver that is too old", () => {
    const described = describeEngineFailure({
      event: "engine_notice",
      code: "CUDA_DRIVER_TOO_OLD",
      gpu_name: "NVIDIA GeForce GTX 1080",
      compute_capability: "6.1",
      driver_cuda_version: "12.1",
      required_cuda_version: "12.8",
    });

    expect(described.detail).toContain("12.8");
    expect(described.detail).toContain("12.1");
    // Updating the driver is the fix; CPU is the workaround.
    expect(described.hint.toLowerCase()).toContain("driver");
  });

  it("uses version-free copy when the numbers are unknown", () => {
    // The runtime path (a child that crashed mid-turn) carries a code and a
    // trace, never the NVML readings -- the copy must still make sense.
    const described = describeEngineFailure({ code: "CUDA_DRIVER_TOO_OLD" });

    expect(described.detail).toBeTruthy();
    expect(described.detail).not.toContain("undefined");
    expect(described.detail).not.toContain("{{");
    expect(described.driverCudaVersion).toBe(null);
    expect(described.requiredCudaVersion).toBe(null);
  });

  it("keeps VRAM exhaustion out of the generic bucket", () => {
    const outOfMemory = describeEngineFailure({ code: "CUDA_OUT_OF_MEMORY" });
    const generic = describeEngineFailure({ code: "CUDA_ERROR" });

    expect(outOfMemory.title).not.toBe(generic.title);
    expect(outOfMemory.hint).not.toBe(generic.hint);
  });

  it("falls back to the generic description for an unknown code", () => {
    const unknown = describeEngineFailure({ code: "SOMETHING_NEW", raw: "a wall of text" });
    const generic = describeEngineFailure({ code: "CUDA_ERROR" });

    expect(unknown.title).toBe(generic.title);
    expect(unknown.code).toBe("SOMETHING_NEW");
    expect(unknown.raw).toBe("a wall of text");
  });

  it("carries the trace under `raw` whichever field the event used", () => {
    // The startup notice calls it `raw`; the chat stream's error event calls it
    // `raw` too, and the chat text lands in `text` -- neither is invented here.
    expect(describeEngineFailure({ code: "CUDA_ERROR", raw: "tail" }).raw).toBe("tail");
    expect(describeEngineFailure({ code: "CUDA_ERROR" }).raw).toBe(null);
  });

  it("returns null for an event with no code at all", () => {
    expect(describeEngineFailure(null)).toBe(null);
    expect(describeEngineFailure({})).toBe(null);
    expect(describeEngineFailure({ t: "error", text: "generic failure" })).toBe(null);
  });
});
