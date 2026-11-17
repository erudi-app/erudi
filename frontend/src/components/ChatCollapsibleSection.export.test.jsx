// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// Export from the history row (#430): the hover menu offers "Export as
// Markdown" next to rename and delete; the work is the shared export
// (utils/conversationExport), and a failure is told, never swallowed.

const { exportMock, navigateMock } = vi.hoisted(() => ({
  exportMock: vi.fn(),
  navigateMock: vi.fn(),
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn(async () => []) },
  apiClient: { get: vi.fn(async () => []) },
  tracedFetch: vi.fn(async () => ({ ok: true })),
}));

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
}));

vi.mock("../utils/conversationExport", () => ({ exportConversation: exportMock }));

import ChatCollapsibleSection from "./ChatCollapsibleSection.jsx";

const items = [
  { id: 7, name: "My chat" },
  { id: 8, name: "Other" },
];

beforeEach(() => {
  exportMock.mockReset();
  exportMock.mockResolvedValue({ status: "saved" });
  vi.spyOn(window, "alert").mockImplementation(() => {});
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ChatCollapsibleSection export", () => {
  it("offers an export on every row, beside rename and delete", () => {
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    expect(screen.getAllByLabelText("Export as Markdown")).toHaveLength(2);
    expect(screen.getAllByLabelText("Rename conversation")).toHaveLength(2);
  });

  it("exports the conversation of the row, without opening it", async () => {
    const onSelect = vi.fn();
    render(<ChatCollapsibleSection title="Previous Chats" items={items} onSelect={onSelect} />);
    fireEvent.click(screen.getAllByLabelText("Export as Markdown")[1]);

    await waitFor(() => expect(exportMock).toHaveBeenCalledTimes(1));
    expect(exportMock.mock.calls[0][0]).toBe(8);
    expect(onSelect).not.toHaveBeenCalled();
    expect(navigateMock).not.toHaveBeenCalled();
    expect(window.alert).not.toHaveBeenCalled();
  });

  it("says so in the error dialog when the export fails, never in a native alert", async () => {
    exportMock.mockRejectedValue(new Error("disk full"));
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    fireEvent.click(screen.getAllByLabelText("Export as Markdown")[0]);

    expect(await screen.findByText("Could not export the conversation: disk full")).toBeTruthy();
    expect(window.alert).not.toHaveBeenCalled();
    fireEvent.click(screen.getAllByRole("button", { name: "Close" })[0]);
    await waitFor(() =>
      expect(screen.queryByText("Could not export the conversation: disk full")).toBeNull()
    );
  });

  describe("from the keyboard", () => {
    const icons = () => [
      screen.getAllByLabelText("Export as Markdown")[0],
      screen.getAllByLabelText("Rename conversation")[0],
      screen.getAllByLabelText("Delete conversation")[0],
    ];

    it("puts the row icons in the tab order", () => {
      render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
      icons().forEach((icon) => expect(icon.getAttribute("tabindex")).toBe("0"));
    });

    it.each(["Enter", " "])("exports on %j without opening the row", async (key) => {
      const onSelect = vi.fn();
      render(<ChatCollapsibleSection title="Previous Chats" items={items} onSelect={onSelect} />);
      fireEvent.keyDown(icons()[0], { key });
      await waitFor(() => expect(exportMock).toHaveBeenCalledTimes(1));
      expect(exportMock.mock.calls[0][0]).toBe(7);
      expect(onSelect).not.toHaveBeenCalled();
    });

    it("renames on Enter and opens the delete confirmation on Space", () => {
      render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
      fireEvent.keyDown(icons()[1], { key: "Enter" });
      expect(screen.getByDisplayValue("My chat")).toBeTruthy();
      fireEvent.keyDown(icons()[2], { key: " " });
      expect(screen.getByText("This action is irreversible.")).toBeTruthy();
    });

    it("ignores the other keys", () => {
      render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
      fireEvent.keyDown(icons()[0], { key: "a" });
      expect(exportMock).not.toHaveBeenCalled();
    });
  });

  it("stays quiet when the user cancels the save dialog", async () => {
    exportMock.mockResolvedValue({ status: "cancelled" });
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    fireEvent.click(screen.getAllByLabelText("Export as Markdown")[0]);
    await waitFor(() => expect(exportMock).toHaveBeenCalled());
    expect(window.alert).not.toHaveBeenCalled();
  });
});
