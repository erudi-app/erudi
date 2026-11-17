// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// #228 — deleting a conversation must fire exactly ONE DELETE request.
// Ownership contract: THIS child owns the mutation. It issues the single DELETE,
// then calls onDelete for post-delete UI only. This test pins the child side of
// the contract (one DELETE + one onDelete per confirmed delete) so a parent's
// onDelete can never re-introduce the original double-fire.

const { tracedFetchMock, navigateMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn(async () => []) },
  apiClient: { get: vi.fn(async () => []) },
  tracedFetch: tracedFetchMock,
}));

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
}));

import ChatCollapsibleSection from "./ChatCollapsibleSection.jsx";
import { resetConversationTurns, trackTurn } from "../utils/conversationTurns";

const items = [{ id: "7", name: "My chat" }];

const deleteCalls = () =>
  tracedFetchMock.mock.calls.filter(([, opts]) => opts?.method === "DELETE");

beforeEach(() => {
  resetConversationTurns();
  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(async () => ({ ok: true }));
});
afterEach(() => {
  cleanup();
});

describe("ChatCollapsibleSection delete (#228)", () => {
  it("fires exactly one DELETE and one onDelete when the delete is confirmed", async () => {
    const onDelete = vi.fn();
    render(<ChatCollapsibleSection title="Previous Chats" items={items} onDelete={onDelete} />);

    // Open the confirm modal, then confirm the delete.
    fireEvent.click(screen.getByLabelText("Delete conversation"));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(onDelete).toHaveBeenCalledTimes(1));

    expect(deleteCalls()).toHaveLength(1);
    expect(String(deleteCalls()[0][0])).toContain("/conversations/7");
    expect(onDelete).toHaveBeenCalledWith("7");
  });

  it("aborts the answer the conversation is streaming, so the delete is not held by it", async () => {
    // The backend answers the DELETE only once the conversation's turn has
    // ended: here, once the answer's fetch is aborted.
    const streaming = new AbortController();
    trackTurn(7, streaming);
    tracedFetchMock.mockImplementation(
      (_url, opts) =>
        new Promise((resolve) => {
          if (opts?.method !== "DELETE") return resolve({ ok: true });
          if (streaming.signal.aborted) return resolve({ ok: true });
          streaming.signal.addEventListener("abort", () => resolve({ ok: true }));
        })
    );
    const onDelete = vi.fn();
    render(<ChatCollapsibleSection title="Previous Chats" items={items} onDelete={onDelete} />);

    fireEvent.click(screen.getByLabelText("Delete conversation"));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(onDelete).toHaveBeenCalledWith("7"));
    expect(streaming.signal.aborted).toBe(true);
  });

  it("sends no second DELETE while the first one is on its way", async () => {
    let answer;
    tracedFetchMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          answer = resolve;
        })
    );
    vi.spyOn(window, "alert").mockImplementation(() => {});
    const onDelete = vi.fn();
    render(<ChatCollapsibleSection title="Previous Chats" items={items} onDelete={onDelete} />);

    fireEvent.click(screen.getByLabelText("Delete conversation"));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(deleteCalls()).toHaveLength(1));
    fireEvent.click(screen.getByLabelText("Delete conversation"));
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    expect(deleteCalls()).toHaveLength(1);
    answer({ ok: true });
    await waitFor(() => expect(onDelete).toHaveBeenCalledTimes(1));
    expect(window.alert).not.toHaveBeenCalled();
  });
});
