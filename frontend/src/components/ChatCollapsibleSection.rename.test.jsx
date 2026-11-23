// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// Rename path of ChatCollapsibleSection: pins the PATCH payload (URL, method,
// headers, body), the onRename callback, and the failure path (alert + no
// onRename). The delete path is pinned separately in
// ChatCollapsibleSection.delete.test.jsx.

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

const items = [{ id: "7", name: "My chat" }];

const patchCalls = () => tracedFetchMock.mock.calls.filter(([, opts]) => opts?.method === "PATCH");

const startEditing = () => {
  fireEvent.click(screen.getByLabelText("Rename conversation"));
  return screen.getByDisplayValue("My chat");
};

beforeEach(() => {
  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(async () => ({ ok: true }));
  vi.spyOn(window, "alert").mockImplementation(() => {});
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ChatCollapsibleSection rename", () => {
  it("fires one PATCH with the trimmed name and calls onRename on Enter", async () => {
    const onRename = vi.fn();
    render(<ChatCollapsibleSection title="Previous Chats" items={items} onRename={onRename} />);

    const input = startEditing();
    fireEvent.change(input, { target: { value: "  Renamed chat  " } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => expect(onRename).toHaveBeenCalledTimes(1));

    expect(patchCalls()).toHaveLength(1);
    const [url, opts] = patchCalls()[0];
    expect(String(url)).toContain("/conversations/7");
    expect(opts.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(opts.body)).toEqual({ name: "Renamed chat" });
    expect(onRename).toHaveBeenCalledWith("7", "Renamed chat");

    // Editing closes after a successful rename.
    await waitFor(() => expect(screen.queryByDisplayValue("Renamed chat")).toBeNull());
  });

  it("alerts and skips onRename when the PATCH fails, and still closes editing", async () => {
    tracedFetchMock.mockImplementation(async () => ({ ok: false, status: 500 }));
    const onRename = vi.fn();
    render(<ChatCollapsibleSection title="Previous Chats" items={items} onRename={onRename} />);

    const input = startEditing();
    fireEvent.change(input, { target: { value: "Renamed chat" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => expect(window.alert).toHaveBeenCalledTimes(1));
    // The status's translated sentence, not a bare "500" (#780).
    expect(String(window.alert.mock.calls[0][0])).toBe(
      "Could not rename conversation, try again: API Error: 500"
    );
    expect(onRename).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.queryByRole("textbox")).toBeNull());
  });

  it("does not PATCH when the new name is blank", () => {
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);

    const input = startEditing();
    fireEvent.change(input, { target: { value: "   " } });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(patchCalls()).toHaveLength(0);
  });

  it("cancels editing on Escape without any PATCH", () => {
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);

    const input = startEditing();
    fireEvent.change(input, { target: { value: "Renamed chat" } });
    fireEvent.keyDown(input, { key: "Escape" });

    expect(patchCalls()).toHaveLength(0);
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.getByText("My chat")).toBeDefined();
  });

  it("closes editing on blur without any PATCH", () => {
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);

    const input = startEditing();
    fireEvent.blur(input);

    expect(patchCalls()).toHaveLength(0);
    expect(screen.queryByRole("textbox")).toBeNull();
  });
});
