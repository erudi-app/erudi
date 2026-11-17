// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

const { getMock, postMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(async () => ({})),
}));

vi.mock("../services/api/client", () => ({
  default: { get: getMock, post: postMock },
}));

import KbEmbeddingBanner from "./KbEmbeddingBanner";

beforeEach(() => {
  getMock.mockReset();
  postMock.mockReset();
  postMock.mockResolvedValue({});
});
afterEach(cleanup);

// "KB present, embedding model missing" surfaced as a clear state in the
// conversation / Arena (#594) — not a blocking modal, not a turn error.
describe("KbEmbeddingBanner", () => {
  it("renders nothing when the assistant has no Knowledge Base", async () => {
    const { container } = render(<KbEmbeddingBanner active={false} />);
    expect(container.firstChild).toBeNull();
    expect(getMock).not.toHaveBeenCalled();
  });

  it("renders nothing while the model is present", async () => {
    getMock.mockResolvedValue({ available: true, downloading: false, error: null });
    const { container } = render(<KbEmbeddingBanner active={true} />);
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    expect(container.firstChild).toBeNull();
  });

  it("shows the missing state with a download action when the model is absent", async () => {
    getMock.mockResolvedValue({ available: false, downloading: false, error: null });
    render(<KbEmbeddingBanner active={true} />);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("embedding model");
    fireEvent.click(screen.getByRole("button"));
    await waitFor(() =>
      expect(postMock).toHaveBeenCalledWith("/knowledge_base/embedding-model/download")
    );
    // The click flips straight to the downloading status line.
    expect(await screen.findByRole("status")).toBeTruthy();
  });

  it("shows the in-flight download as a status, not an alert", async () => {
    getMock.mockResolvedValue({ available: false, downloading: true, error: null });
    render(<KbEmbeddingBanner active={true} />);
    expect(await screen.findByRole("status")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("re-checks the status when the window regains focus", async () => {
    getMock.mockResolvedValue({ available: true, downloading: false, error: null });
    render(<KbEmbeddingBanner active={true} />);
    await waitFor(() => expect(getMock).toHaveBeenCalledTimes(1));
    // The model was deleted while the app was in the background: coming back
    // must surface the state without a remount.
    getMock.mockResolvedValue({ available: false, downloading: false, error: null });
    fireEvent(window, new Event("focus"));
    await screen.findByRole("alert");
    expect(getMock).toHaveBeenCalledTimes(2);
  });

  it("shows the failed download with a retry action", async () => {
    getMock.mockResolvedValue({
      available: false,
      downloading: false,
      error: "no internet connection — the embedding model could not be downloaded",
    });
    render(<KbEmbeddingBanner active={true} />);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("no internet connection");
    expect(screen.getByRole("button").textContent).toContain("Retry");
  });
});
