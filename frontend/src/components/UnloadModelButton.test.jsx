// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// The Unload button of the conversation header's settings panel (#619): it
// asks the backend which model is loaded when it appears (the panel opens),
// names that model whichever conversation loaded it, and is absent when
// nothing is loaded. Pressing it unloads the model and asks again; a busy
// engine (409) is refused with a short message and nothing is interrupted.

const { getMock, postMock } = vi.hoisted(() => ({ getMock: vi.fn(), postMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: getMock, post: postMock },
}));

import UnloadModelButton from "./UnloadModelButton.jsx";

const LOADED = { llm_id: 3, name: "Qwen3-4B" };
const BUSY = "The model is busy; try again in a moment.";

const httpError = (status, type) =>
  Object.assign(new Error(`HTTP ${status}`), {
    status,
    data: type ? { error: { type } } : {},
  });

beforeEach(() => {
  getMock.mockReset();
  postMock.mockReset();
});

afterEach(cleanup);

describe("UnloadModelButton", () => {
  it("asks which model is loaded when it appears and names it", async () => {
    getMock.mockResolvedValue(LOADED);
    render(<UnloadModelButton />);
    expect(await screen.findByRole("button", { name: "Unload Qwen3-4B" })).toBeTruthy();
    expect(getMock).toHaveBeenCalledTimes(1);
    expect(getMock).toHaveBeenCalledWith("/llms/loaded");
  });

  it("shows nothing while nothing is loaded", async () => {
    getMock.mockResolvedValue(null);
    const { container } = render(<UnloadModelButton />);
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    expect(container.textContent).toBe("");
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("shows nothing when the backend cannot say", async () => {
    getMock.mockRejectedValue(httpError(500));
    const { container } = render(<UnloadModelButton />);
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    expect(container.textContent).toBe("");
  });

  it("unloads the model, asks again and disappears", async () => {
    getMock.mockResolvedValueOnce(LOADED).mockResolvedValueOnce(null);
    postMock.mockResolvedValue(undefined);
    render(<UnloadModelButton />);
    fireEvent.click(await screen.findByRole("button", { name: "Unload Qwen3-4B" }));
    await waitFor(() => expect(screen.queryByRole("button")).toBeNull());
    expect(postMock).toHaveBeenCalledWith("/llms/loaded/unload", { llm_id: 3 });
    expect(getMock).toHaveBeenCalledTimes(2);
  });

  it("is refused while the model is busy, and stays", async () => {
    getMock.mockResolvedValue(LOADED);
    postMock.mockRejectedValue(httpError(409));
    render(<UnloadModelButton />);
    fireEvent.click(await screen.findByRole("button", { name: "Unload Qwen3-4B" }));
    expect((await screen.findByRole("alert")).textContent).toBe(BUSY);
    expect(screen.getByRole("button", { name: "Unload Qwen3-4B" })).toBeTruthy();
  });

  it("says when the unload failed for another reason, then asks again", async () => {
    getMock.mockResolvedValue(LOADED);
    postMock.mockRejectedValue(httpError(500));
    render(<UnloadModelButton />);
    fireEvent.click(await screen.findByRole("button", { name: "Unload Qwen3-4B" }));
    expect((await screen.findByRole("alert")).textContent).toBe("The model could not be unloaded.");
    await waitFor(() => expect(getMock).toHaveBeenCalledTimes(2));
  });

  it("is disabled while the request runs", async () => {
    getMock.mockResolvedValue(LOADED);
    let finish;
    postMock.mockReturnValue(
      new Promise((resolve) => {
        finish = resolve;
      })
    );
    render(<UnloadModelButton />);
    const button = await screen.findByRole("button", { name: "Unload Qwen3-4B" });
    fireEvent.click(button);
    await waitFor(() => expect(button.disabled).toBe(true));
    getMock.mockResolvedValue(null);
    finish();
    await waitFor(() => expect(screen.queryByRole("button")).toBeNull());
  });

  it("clears the busy message once a later press succeeds", async () => {
    getMock.mockResolvedValue(LOADED);
    postMock.mockRejectedValueOnce(httpError(409)).mockResolvedValueOnce(undefined);
    render(<UnloadModelButton />);
    const button = await screen.findByRole("button", { name: "Unload Qwen3-4B" });
    fireEvent.click(button);
    await screen.findByRole("alert");
    getMock.mockResolvedValue(null);
    fireEvent.click(button);
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    await waitFor(() => expect(screen.queryByRole("button")).toBeNull());
  });

  it("unloads only the model it names: another one loaded since is left alone", async () => {
    // The button read Qwen3-4B; a model switch and a turn loaded Gemma since.
    getMock.mockResolvedValueOnce(LOADED).mockResolvedValue({ llm_id: 8, name: "Gemma" });
    postMock.mockRejectedValue(httpError(409, "LOADED_MODEL_CHANGED"));
    render(<UnloadModelButton />);
    fireEvent.click(await screen.findByRole("button", { name: "Unload Qwen3-4B" }));
    expect(postMock).toHaveBeenCalledWith("/llms/loaded/unload", { llm_id: 3 });
    expect((await screen.findByRole("alert")).textContent).toBe("Another model is loaded now.");
    // It reads the loaded model again and names it.
    expect(await screen.findByRole("button", { name: "Unload Gemma" })).toBeTruthy();
  });

  it("asks again when the page says a turn ended or the model changed", async () => {
    getMock.mockResolvedValue(null);
    const { rerender } = render(<UnloadModelButton refreshKey="0:3" />);
    await waitFor(() => expect(getMock).toHaveBeenCalledTimes(1));
    expect(screen.queryByRole("button")).toBeNull();

    getMock.mockResolvedValue(LOADED);
    rerender(<UnloadModelButton refreshKey="0:3" />);
    expect(getMock).toHaveBeenCalledTimes(1);
    // One more turn ended.
    rerender(<UnloadModelButton refreshKey="1:3" />);
    expect(await screen.findByRole("button", { name: "Unload Qwen3-4B" })).toBeTruthy();
    expect(getMock).toHaveBeenCalledTimes(2);
  });

  it("drops a refusal once the page says something changed", async () => {
    getMock.mockResolvedValue(LOADED);
    postMock.mockRejectedValue(httpError(409));
    const { rerender } = render(<UnloadModelButton refreshKey="0:3" />);
    fireEvent.click(await screen.findByRole("button", { name: "Unload Qwen3-4B" }));
    expect((await screen.findByRole("alert")).textContent).toBe(BUSY);
    rerender(<UnloadModelButton refreshKey="1:3" />);
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    expect(screen.getByRole("button", { name: "Unload Qwen3-4B" })).toBeTruthy();
  });
});
