import { describe, it, expect } from "vitest";
import { formatBytes } from "./formatBytes";

// Human-readable size for the memory warning (1.1.2): binary units, one
// decimal above 1 GiB, integers below.
describe("formatBytes", () => {
  it("formats gigabytes with one decimal", () => {
    expect(formatBytes(4 * 1024 ** 3)).toBe("4.0 GB");
    expect(formatBytes(4.75 * 1024 ** 3)).toBe("4.7 GB");
  });

  it("formats megabytes as integers", () => {
    expect(formatBytes(12 * 1024 ** 2)).toBe("12 MB");
  });

  it("formats small sizes as kilobytes", () => {
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(10)).toBe("1 KB");
    expect(formatBytes(0)).toBe("0 KB");
  });

  it("answers null for non-finite input", () => {
    expect(formatBytes(null)).toBeNull();
    expect(formatBytes(undefined)).toBeNull();
    expect(formatBytes(-5)).toBeNull();
    expect(formatBytes(NaN)).toBeNull();
  });
});
