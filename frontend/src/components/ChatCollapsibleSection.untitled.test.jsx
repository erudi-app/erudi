// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import i18n from "../i18n";

// The backend stores one neutral name for a conversation with no title; the
// sidebar shows it in the interface language.

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn(async () => []) },
  apiClient: { get: vi.fn(async () => []) },
  tracedFetch: vi.fn(),
}));
vi.mock("react-router-dom", () => ({ useNavigate: () => vi.fn() }));

import ChatCollapsibleSection from "./ChatCollapsibleSection.jsx";

afterEach(async () => {
  cleanup();
  await i18n.changeLanguage("en");
});

describe("ChatCollapsibleSection untitled conversations", () => {
  it("shows the stored default name in the interface language", async () => {
    await i18n.changeLanguage("fr");
    render(
      <ChatCollapsibleSection
        title="Previous Chats"
        items={[
          { id: "1", name: "New Conversation" },
          { id: "2", name: "Photosynthèse" },
        ]}
      />
    );
    expect(screen.getByText("Nouvelle conversation")).toBeTruthy();
    expect(screen.queryByText("New Conversation")).toBeNull();
    expect(screen.getByText("Photosynthèse")).toBeTruthy();
  });

  it("opens the rename input empty for the default name, with the translated name as placeholder", async () => {
    await i18n.changeLanguage("fr");
    render(
      <ChatCollapsibleSection
        title="Previous Chats"
        items={[{ id: "1", name: "New Conversation" }]}
      />
    );
    fireEvent.click(screen.getByLabelText(i18n.t("chat:history.renameConversation")));
    const input = screen.getByPlaceholderText("Nouvelle conversation");
    expect(input.value).toBe("");
  });

  it("seeds the rename input with a real title", async () => {
    render(
      <ChatCollapsibleSection title="Previous Chats" items={[{ id: "2", name: "Photosynthèse" }]} />
    );
    fireEvent.click(screen.getByLabelText(i18n.t("chat:history.renameConversation")));
    expect(screen.getByDisplayValue("Photosynthèse")).toBeTruthy();
  });
});
