import { describe, it, expect } from "vitest";
import {
  deriveDownloadPhase,
  downloadStalledMessage,
  DOWNLOAD_CANCELLED,
  DOWNLOAD_STALLED,
} from "./downloadStatus";

// The widget renders one of a handful of phases; the context derives them from
// the raw (status, progress, errorMessage) triple it tracks (#292).
describe("deriveDownloadPhase (#292)", () => {
  it("is queued before the job runs", () => {
    expect(deriveDownloadPhase({ status: "idle", progress: 0 })).toBe("queued");
    expect(deriveDownloadPhase({ status: "pending", progress: 0 })).toBe("queued");
  });

  it("splits a running job into downloading and finalizing at 100%", () => {
    expect(deriveDownloadPhase({ status: "running", progress: 42 })).toBe("downloading");
    expect(deriveDownloadPhase({ status: "running", progress: 99.9 })).toBe("downloading");
    expect(deriveDownloadPhase({ status: "running", progress: 100 })).toBe("finalizing");
  });

  it("maps the terminal statuses one-to-one", () => {
    expect(deriveDownloadPhase({ status: "completed", progress: 100 })).toBe("completed");
    expect(deriveDownloadPhase({ status: "failed", progress: 10 })).toBe("failed");
    expect(deriveDownloadPhase({ status: DOWNLOAD_CANCELLED, progress: 10 })).toBe("cancelled");
    expect(deriveDownloadPhase({ status: DOWNLOAD_STALLED, progress: 100 })).toBe("stalled");
  });

  it("treats a client-side error (failed start, poll error) as a failure", () => {
    expect(deriveDownloadPhase({ status: "pending", progress: 0, errorMessage: "boom" })).toBe(
      "failed"
    );
    expect(deriveDownloadPhase({ status: "running", progress: 30, errorMessage: "boom" })).toBe(
      "failed"
    );
  });

  it("keeps a cancel that failed to reach the backend a cancellation, not a failure", () => {
    expect(
      deriveDownloadPhase({ status: DOWNLOAD_CANCELLED, progress: 0, errorMessage: "500" })
    ).toBe("cancelled");
  });
});

describe("DOWNLOAD_CANCELLED", () => {
  it("matches the status string the poll/cancel path uses", () => {
    expect(DOWNLOAD_CANCELLED).toBe("cancelled");
  });
});

describe("DOWNLOAD_STALLED (#315)", () => {
  it("is a distinct client-side status, never one of the backend terminal states", () => {
    expect(DOWNLOAD_STALLED).toBe("stalled");
    expect(DOWNLOAD_STALLED).not.toBe(DOWNLOAD_CANCELLED);
  });

  it("maps to the finalizing message, NOT the generic download-failed one", () => {
    // The bytes are on disk; only the finalization bookkeeping did not finish
    // (#291). Reporting "Download failed. Please try again." would be wrong and
    // would push the user into re-downloading gigabytes they already have.
    expect(downloadStalledMessage()).toMatch(/files have been saved/i);
  });
});
