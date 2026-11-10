// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// A KB assistant whose base model was deleted survives with
// `weights_available === false` (#376). Its Installed card already disables
// Chat and offers Re-bind; the Chat page must agree: /model does not offer it,
// it is never auto-selected (neither as the default nor through `?model=`),
// and with nothing selectable the header asks for a model and the composer
// stays disabled instead of creating a conversation that can only fail.

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  tracedFetch: tracedFetchMock,
}));

vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: vi.fn(), completionCount: 0 }),
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/GradientBox", () => ({ default: ({ children }) => <div>{children}</div> }));
// The composer stub exposes the `disabled` prop the page passes it.
vi.mock("../components/QuestionInput", () => ({
  default: ({ disabled }) => <div data-testid="composer" data-disabled={String(disabled)} />,
}));
vi.mock("../components/modals/ErrorModal", () => ({ default: () => null }));

import apiClient from "../services/api/client";
import ChatPage from "./ChatPage.jsx";

const ORPHAN = {
  id: 9,
  name: "Nimbus Assistant",
  kb_id: 3,
  is_attached_to_kb: true,
  weights_available: false,
};
const HEALTHY = { id: 42, name: "Beta Model", weights_available: true };

const renderAt = (search = "") =>
  render(
    <MemoryRouter initialEntries={[`/erudi/chat${search}`]}>
      <ChatPage />
    </MemoryRouter>
  );

const useModels = (models) => {
  apiClient.get.mockImplementation(async (path) => (path === "/llms/local" ? models : []));
};

const openModelCommand = async () => {
  fireEvent.click(await screen.findByRole("button", { name: "Conversation settings" }));
  fireEvent.click(screen.getByRole("button", { name: /^Model/ }));
  return screen.findByRole("dialog");
};

beforeEach(() => {
  apiClient.get.mockReset();
});
afterEach(() => {
  cleanup();
});

describe("ChatPage picker and a weights-missing assistant (#376)", () => {
  it("is not offered by /model", async () => {
    useModels([ORPHAN, HEALTHY]);
    renderAt();
    await screen.findByTitle("Beta Model");

    const dialog = await openModelCommand();
    expect(within(dialog).queryByText("Nimbus Assistant")).toBeNull();
    expect(within(dialog).getByRole("button", { name: /Beta Model/ })).toBeTruthy();
  });

  it("skips the orphan and preselects the next selectable model", async () => {
    useModels([ORPHAN, HEALTHY]);
    renderAt();

    const selected = await screen.findByTitle("Beta Model");
    expect(selected.textContent).toBe("Beta Model");
    expect(screen.getByTestId("composer").getAttribute("data-disabled")).toBe("false");
  });

  it("leaves nothing selected and the composer disabled when the orphan is the only model", async () => {
    useModels([ORPHAN]);
    renderAt();

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Please select a model" }).textContent).toBe(
        "Please select a model"
      )
    );
    expect(screen.queryByTitle("Nimbus Assistant")).toBeNull();
    expect(screen.getByTestId("composer").getAttribute("data-disabled")).toBe("true");
    const dialog = await openModelCommand();
    expect(within(dialog).getByText("No installed model can be used right now.")).toBeTruthy();
  });

  it("ignores a ?model= deep link that points at the orphan", async () => {
    useModels([ORPHAN, HEALTHY]);
    renderAt("?model=9");

    const selected = await screen.findByTitle("Beta Model");
    expect(selected.textContent).toBe("Beta Model");
    await waitFor(() => expect(screen.queryByTitle("Nimbus Assistant")).toBeNull());
  });
});
