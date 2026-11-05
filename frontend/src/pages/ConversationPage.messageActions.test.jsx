// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// #136 — message action buttons:
// D: the copy button must put the CLEANED display text on the clipboard
//    (internal [image]/[image_path:…] markers stripped), not the raw content.
// F: the star toggle is optimistic — it flips immediately on click and rolls
//    back SILENTLY (owner decision: no error UI) when the request fails,
//    whether the server answers non-ok or the network rejects.

const { tracedFetchMock, navigateMock, locationMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
  // Stable identities: the page's mount effect lists navigate/location values
  // in its dependency array, so fresh mocks per render would loop it forever.
  locationMock: { pathname: "/conversation/7", state: null },
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn(async () => []) },
  apiClient: { get: vi.fn(async () => []) },
  tracedFetch: tracedFetchMock,
}));

vi.mock("react-router-dom", () => ({
  useParams: () => ({ id: "7" }),
  useNavigate: () => navigateMock,
  useLocation: () => locationMock,
}));

// Stub the heavy children — only the message list and its action buttons matter.
vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/QuestionInput", () => ({ default: () => null }));
vi.mock("../components/ConversationHeader", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({ default: () => null }));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";

const RAW_WITH_MARKERS = "[image_path:/a/b.png][image]Describe the picture";
const CLEANED = "Describe the picture";

const messages = [{ id: 101, sender: "user", content: RAW_WITH_MARKERS, starred: false }];

const conversationDetail = {
  id: 7,
  llm_id: 1,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
};

// Route map for tracedFetch; `starResponder` decides the star/unstar outcome.
const makeRouteFetch =
  (starResponder) =>
  async (url, opts = {}) => {
    const u = String(url);
    if (u.includes("star_message")) return starResponder(u, opts);
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
    if (u.includes("fetch_messages")) return { ok: true, json: async () => messages };
    return { ok: true, json: async () => [] };
  };

const renderAndSettle = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
  await screen.findByTitle("Star message");
};

const starIcon = () => screen.getByTitle("Star message").querySelector("svg");
const isStarredInUI = () => starIcon().getAttribute("fill") === "currentColor";

// The rollback must be silent (owner decision): no toast, alert, or error text.
const expectNoErrorSurface = () => {
  expect(document.querySelector('[role="alert"]')).toBeNull();
  expect(screen.queryByText(/error|failed/i)).toBeNull();
};

// jsdom does not implement Element.scrollTo (used by the auto-scroll effect).
beforeEach(() => {
  Element.prototype.scrollTo = () => {};
  tracedFetchMock.mockReset();
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async () => messages);
  tracedFetchMock.mockImplementation(makeRouteFetch(async () => ({ ok: true })));
});
afterEach(() => {
  cleanup();
});

describe("copy button (#136 D)", () => {
  it("copies the cleaned display text, not the raw content with markers", async () => {
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    await renderAndSettle();

    fireEvent.click(screen.getByTitle("Copy message"));

    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText).toHaveBeenCalledWith(CLEANED);
    // Flush the copied-checkmark state update queued by the .then().
    await act(async () => {});
  });
});

describe("star toggle (#136 F)", () => {
  it("flips the star immediately, before the request resolves", async () => {
    let resolveStar;
    const pending = new Promise((resolve) => {
      resolveStar = resolve;
    });
    tracedFetchMock.mockImplementation(makeRouteFetch(() => pending));

    await renderAndSettle();
    expect(isStarredInUI()).toBe(false);

    fireEvent.click(screen.getByTitle("Star message"));

    // Optimistic: the UI flips before the server has answered.
    expect(isStarredInUI()).toBe(true);

    resolveStar({ ok: true });
    await act(async () => {});
    expect(isStarredInUI()).toBe(true);
  });

  it("reverts silently when the server answers non-ok (HTTP 500)", async () => {
    tracedFetchMock.mockImplementation(makeRouteFetch(async () => ({ ok: false, status: 500 })));

    await renderAndSettle();

    fireEvent.click(screen.getByTitle("Star message"));
    expect(isStarredInUI()).toBe(true); // optimistic flip first

    await waitFor(() => expect(isStarredInUI()).toBe(false)); // rolled back
    expectNoErrorSurface();
  });

  it("reverts silently when the request rejects at the network level", async () => {
    tracedFetchMock.mockImplementation(
      makeRouteFetch(async () => {
        throw new TypeError("network down");
      })
    );

    await renderAndSettle();

    fireEvent.click(screen.getByTitle("Star message"));
    expect(isStarredInUI()).toBe(true); // optimistic flip first

    await waitFor(() => expect(isStarredInUI()).toBe(false)); // rolled back
    expectNoErrorSurface();
  });
});
