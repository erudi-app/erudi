// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, waitFor, act } from "@testing-library/react";

// #136 — once a pasted image is persisted, it is stored with a normal
// [image_path:...] marker (same shape a file attachment produces). On reload the
// page restores it from disk via window.fsAPI.readImageAsDataURL and renders it
// as a thumbnail, exactly like any path-attached image.

const { tracedFetchMock, navigateMock, locationMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
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

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/QuestionInput", () => ({ default: () => null }));
vi.mock("../components/HeaderBar", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({ default: () => null }));
vi.mock("../components/modals/CustomizePromptModal", () => ({ default: () => null }));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";

const PASTED_PATH = "C:\\Users\\me\\AppData\\Local\\erudi\\pasted-images\\paste-1.png";
const RESTORED_DATA_URL = "data:image/png;base64,UkVTVE9SRUQ=";

// A persisted pasted image carries a bare path marker, no [image] fallback.
const messages = [
  { id: 101, sender: "user", content: `[image_path:${PASTED_PATH}] Describe this`, starred: false },
];

const conversationDetail = {
  id: 7,
  llm_id: 1,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
};

beforeEach(() => {
  Element.prototype.scrollTo = () => {};
  window.fsAPI = { readImageAsDataURL: vi.fn(async () => RESTORED_DATA_URL) };
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async () => messages);
  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(async (url) => {
    const u = String(url);
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
    return { ok: true, json: async () => [] };
  });
});

afterEach(() => {
  cleanup();
  delete window.fsAPI;
});

describe("ConversationPage image restore with an encoded path", () => {
  it("reads back the DECODED path when the marker carries ] or %", async () => {
    // The backend percent-encodes "]" and "%" so they cannot end the marker;
    // the page must hand fsAPI the real path, not the encoded one.
    apiClient.get.mockImplementation(async () => [
      {
        id: 202,
        sender: "user",
        content: "Describe this [image_path:/photos/[2026%5D 100%25/shot.png]",
        starred: false,
      },
    ]);

    render(<ConversationPage />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    await act(async () => {});

    await waitFor(() =>
      expect(window.fsAPI.readImageAsDataURL).toHaveBeenCalledWith("/photos/[2026] 100%/shot.png")
    );
  });
});

describe("ConversationPage pasted-image restore (#136)", () => {
  it("restores a persisted [image_path:...] marker as an image thumbnail", async () => {
    render(<ConversationPage />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    await act(async () => {});

    // The stored path was read back from disk...
    await waitFor(() => expect(window.fsAPI.readImageAsDataURL).toHaveBeenCalledWith(PASTED_PATH));

    // ...and rendered as a normal image thumbnail (not a bare placeholder).
    const img = await screen.findByAltText("attachment 1");
    expect(img.getAttribute("src")).toBe(RESTORED_DATA_URL);
    expect(screen.queryByText(/image attachment/)).toBeNull();
  });
});
