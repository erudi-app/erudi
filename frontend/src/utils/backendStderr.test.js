import { describe, it, expect } from "vitest";
import { classifyStderrLine, isBenignStderrLine } from "./backendStderr.js";

describe("isBenignStderrLine", () => {
  it("treats CPU-build GPU/CUDA/NVML noise as benign", () => {
    expect(isBenignStderrLine("WARNING: CUDA not available, using CPU")).toBe(true);
    expect(isBenignStderrLine("pynvml.NVMLError: driver not loaded")).toBe(true);
  });

  it("treats routine database/sqlalchemy log lines as benign", () => {
    expect(isBenignStderrLine("INFO sqlalchemy.engine connected to database erudi")).toBe(true);
    expect(isBenignStderrLine("psycopg: connection established")).toBe(true);
  });

  it("returns false for empty or ordinary lines", () => {
    expect(isBenignStderrLine("")).toBe(false);
    expect(isBenignStderrLine("Uvicorn running on http://127.0.0.1:8765")).toBe(false);
  });
});

describe("classifyStderrLine", () => {
  it("never marks a line fatal — the backend's JSON events decide failure", () => {
    const lines = [
      "WARNING: CUDA not available, using CPU",
      "pynvml.NVMLError",
      "mlx.core metal unavailable",
      "sqlalchemy.engine database ready",
      "No module named 'foo'",
      "Uvicorn running",
    ];
    for (const line of lines) {
      const result = classifyStderrLine(line);
      if (result) expect(result.fatal).toBe(false);
    }
  });

  it("extracts a missing-module hint (non-fatal)", () => {
    const r = classifyStderrLine("ModuleNotFoundError: No module named 'numpy'");
    expect(r).toEqual({
      code: "MISSING_DEPENDENCY",
      message: "Missing Python module: numpy",
      fatal: false,
    });
  });

  it("classifies benign GPU noise as a non-fatal diagnostic", () => {
    const r = classifyStderrLine("CUDA not available");
    expect(r.code).toBe("GPU_DIAGNOSTIC");
    expect(r.fatal).toBe(false);
  });

  it("returns null for unremarkable lines", () => {
    expect(classifyStderrLine("Uvicorn running on http://127.0.0.1:8765")).toBeNull();
    expect(classifyStderrLine("")).toBeNull();
  });
});
