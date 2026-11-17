// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// Search across the history (#429): the field in the sidebar turns the list into
// the conversations the backend finds for the typed text, after a pause, and
// typing nothing brings the list back.

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

const items = [
  { id: 1, name: "Cooking" },
  { id: 2, name: "Airships" },
  { id: 3, name: "Chess" },
];

const searchCalls = () =>
  tracedFetchMock.mock.calls.filter(([url]) => String(url).includes("/conversations/search"));

const field = () => screen.getByRole("searchbox", { name: "Search conversations" });
const type = (value) => fireEvent.change(field(), { target: { value } });

const answer = (rows) =>
  tracedFetchMock.mockImplementation(async () => ({ ok: true, json: async () => rows }));

beforeEach(() => {
  tracedFetchMock.mockReset();
  answer([]);
  vi.spyOn(window, "alert").mockImplementation(() => {});
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ChatCollapsibleSection search", () => {
  it("shows the history untouched until something is typed", () => {
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    expect(field().value).toBe("");
    ["Cooking", "Airships", "Chess"].forEach((name) => expect(screen.getByText(name)).toBeTruthy());
    expect(searchCalls()).toHaveLength(0);
  });

  it("replaces the list with the matches, asking the backend with the typed text", async () => {
    answer([{ id: 2, name: "Airships" }]);
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    type("zeppelin & co");

    await waitFor(() => expect(screen.queryByText("Cooking")).toBeNull());
    expect(screen.getByText("Airships")).toBeTruthy();
    expect(screen.queryByText("Chess")).toBeNull();
    expect(searchCalls()).toHaveLength(1);
    expect(String(searchCalls()[0][0])).toContain("/conversations/search?q=zeppelin%20%26%20co");
  });

  it("waits for a pause in the typing: one request for a burst", async () => {
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    type("z");
    type("ze");
    type("zep");
    expect(searchCalls()).toHaveLength(0);
    await waitFor(() => expect(searchCalls()).toHaveLength(1));
    expect(String(searchCalls()[0][0])).toContain("q=zep");
  });

  it("says so when nothing matches", async () => {
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    type("nothing like this");
    expect(await screen.findByText("No conversation matches.")).toBeTruthy();
    expect(screen.queryByText("Cooking")).toBeNull();
  });

  it("brings the whole list back when the field is emptied, without a request", async () => {
    answer([{ id: 2, name: "Airships" }]);
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    type("air");
    await waitFor(() => expect(screen.queryByText("Cooking")).toBeNull());

    type("");
    expect(screen.getByText("Cooking")).toBeTruthy();
    expect(screen.getByText("Chess")).toBeTruthy();
    expect(searchCalls()).toHaveLength(1);
  });

  it("brings the list back with the clear button and with Escape", async () => {
    answer([{ id: 2, name: "Airships" }]);
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    type("air");
    await waitFor(() => expect(screen.queryByText("Cooking")).toBeNull());

    fireEvent.click(screen.getByRole("button", { name: "Clear search" }));
    expect(field().value).toBe("");
    expect(screen.getByText("Cooking")).toBeTruthy();

    type("air");
    await waitFor(() => expect(screen.queryByText("Cooking")).toBeNull());
    fireEvent.keyDown(field(), { key: "Escape" });
    expect(field().value).toBe("");
    expect(screen.getByText("Cooking")).toBeTruthy();
  });

  it("treats a blank query as no search", async () => {
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    type("   ");
    await new Promise((resolve) => setTimeout(resolve, 450));
    expect(searchCalls()).toHaveLength(0);
    expect(screen.getByText("Cooking")).toBeTruthy();
  });

  it("ignores the answer of a query the user has typed over", async () => {
    let releaseFirst;
    tracedFetchMock
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            releaseFirst = () =>
              resolve({ ok: true, json: async () => [{ id: 11, name: "Cookbook" }] });
          })
      )
      .mockImplementationOnce(async () => ({
        ok: true,
        json: async () => [{ id: 12, name: "Airship log" }],
      }));
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    type("coo");
    await waitFor(() => expect(searchCalls()).toHaveLength(1));
    type("air");
    await screen.findByText("Airship log");
    releaseFirst();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.getByText("Airship log")).toBeTruthy();
    expect(screen.queryByText("Cookbook")).toBeNull();
  });

  it("tells the user when the search fails, and keeps the typed text", async () => {
    tracedFetchMock.mockImplementation(async () => ({ ok: false, status: 500 }));
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    type("air");
    expect(await screen.findByText("The search failed. Try again.")).toBeTruthy();
    expect(field().value).toBe("air");
  });

  it("opens a found conversation like any row", async () => {
    answer([{ id: 2, name: "Airships" }]);
    const onSelect = vi.fn();
    render(<ChatCollapsibleSection title="Previous Chats" items={items} onSelect={onSelect} />);
    type("air");
    fireEvent.click(await screen.findByText("Airships"));
    expect(onSelect).toHaveBeenCalledWith(2);
    expect(navigateMock).toHaveBeenCalled();
  });

  it("drops a deleted conversation from the results", async () => {
    answer([
      { id: 2, name: "Airships" },
      { id: 4, name: "Airplanes" },
    ]);
    render(<ChatCollapsibleSection title="Previous Chats" items={items} onDelete={vi.fn()} />);
    type("air");
    await screen.findByText("Airplanes");
    tracedFetchMock.mockImplementation(async () => ({ ok: true }));

    fireEvent.click(screen.getAllByLabelText("Delete conversation")[1]);
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(screen.queryByText("Airplanes")).toBeNull());
    expect(screen.getByText("Airships")).toBeTruthy();
  });
});
