// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// The header controls of the sidebar section are real buttons (#766): the
// header toggle, refresh, new chat and the search clear cross are reachable
// with Tab and work with Enter / Space, and the two icons never toggle the
// section.

const { tracedFetchMock, navigateMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn(async () => []) },
  apiClient: { get: vi.fn(async () => []) },
  tracedFetch: tracedFetchMock,
}));
vi.mock("react-router-dom", () => ({ useNavigate: () => navigateMock }));

import ChatCollapsibleSection from "./ChatCollapsibleSection.jsx";

const items = [{ id: "7", name: "My chat" }];

beforeEach(() => {
  tracedFetchMock.mockReset();
  navigateMock.mockReset();
});
afterEach(cleanup);

const toggleButton = () => screen.getByRole("button", { name: /Previous Chats/ });

describe("ChatCollapsibleSection keyboard access (#766)", () => {
  it("renders the header toggle, refresh and new chat as buttons", () => {
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    expect(screen.getByRole("button", { name: "New chat" }).tagName).toBe("BUTTON");
    expect(screen.getByRole("button", { name: "Refresh conversations" }).tagName).toBe("BUTTON");
    expect(toggleButton().tagName).toBe("BUTTON");
  });

  it("reaches the controls with Tab and triggers new chat with Enter without collapsing", async () => {
    const user = userEvent.setup();
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    await user.tab();
    expect(document.activeElement).toBe(toggleButton());
    await user.tab();
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "Refresh conversations" })
    );
    await user.tab();
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "New chat" }));
    await user.keyboard("{Enter}");
    expect(navigateMock).toHaveBeenCalledWith("/erudi/chat");
    expect(toggleButton().getAttribute("aria-expanded")).toBe("true");
  });

  it("refreshes with Space without collapsing", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn(async () => {});
    render(<ChatCollapsibleSection title="Previous Chats" items={items} onRefresh={onRefresh} />);
    screen.getByRole("button", { name: "Refresh conversations" }).focus();
    await user.keyboard(" ");
    await waitFor(() => expect(onRefresh).toHaveBeenCalledTimes(1), { timeout: 2000 });
    expect(toggleButton().getAttribute("aria-expanded")).toBe("true");
  });

  it("toggles the section with aria-expanded", async () => {
    const user = userEvent.setup();
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    const toggle = toggleButton();
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    toggle.focus();
    await user.keyboard("{Enter}");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
  });

  it("takes every control out of reach while disabled (an answer is streaming)", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <ChatCollapsibleSection title="Previous Chats" items={items} disabled />
    );
    expect(container.firstChild.hasAttribute("inert")).toBe(true);
    for (const name of [/Previous Chats/, "Refresh conversations", "New chat"]) {
      expect(screen.getByRole("button", { name }).disabled).toBe(true);
    }
    await user.click(screen.getByRole("button", { name: "New chat" }));
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it("clears the search with a button", async () => {
    const user = userEvent.setup();
    render(<ChatCollapsibleSection title="Previous Chats" items={items} />);
    const input = screen.getByRole("searchbox");
    await user.type(input, "abc");
    const clear = screen.getByRole("button", { name: "Clear search" });
    expect(clear.tagName).toBe("BUTTON");
    await user.click(clear);
    expect(input.value).toBe("");
  });
});
