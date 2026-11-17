// @vitest-environment jsdom
import React, { useEffect } from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, cleanup, act } from "@testing-library/react";
import { MemoryRouter, useNavigate } from "react-router-dom";

// #428: an ephemeral conversation is deleted when the conversation id in the
// route changes, and only then: never on a re-render of the same route.

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));
vi.mock("../services/api/client", () => ({ tracedFetch: tracedFetchMock }));

import EphemeralConversationGuard from "./EphemeralConversationGuard";
import {
  markEphemeral,
  resetEphemeralConversations,
  trackTurn,
} from "../utils/ephemeralConversations";

let navigate;
const keepNavigate = (fn) => {
  navigate = fn;
};
// Hands the router's navigate to the test.
function NavigateHandle() {
  const routerNavigate = useNavigate();
  useEffect(() => keepNavigate(routerNavigate), [routerNavigate]);
  return null;
}

const renderAt = (path) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <EphemeralConversationGuard />
      <NavigateHandle />
    </MemoryRouter>
  );

const deletes = () =>
  tracedFetchMock.mock.calls.filter(([, options]) => options?.method === "DELETE");

beforeEach(() => {
  resetEphemeralConversations();
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue({ ok: true, status: 200 });
});
afterEach(cleanup);

describe("EphemeralConversationGuard", () => {
  it("deletes an ephemeral conversation when another conversation opens", async () => {
    markEphemeral(5);
    renderAt("/erudi/conversations/5");

    await act(async () => navigate("/erudi/conversations/6"));

    expect(deletes()).toHaveLength(1);
    expect(String(deletes()[0][0])).toMatch(/\/conversations\/5$/);
  });

  it.each([["/erudi/chat"], ["/erudi/settings"]])(
    "deletes it when the user goes to %s",
    async (path) => {
      markEphemeral("5");
      renderAt("/erudi/conversations/5");

      await act(async () => navigate(path));

      expect(deletes()).toHaveLength(1);
    }
  );

  it("does nothing on a re-render or a navigation to the same conversation", async () => {
    markEphemeral(5);
    const view = renderAt("/erudi/conversations/5");

    view.rerender(
      <MemoryRouter initialEntries={["/erudi/conversations/5"]}>
        <EphemeralConversationGuard />
        <NavigateHandle />
      </MemoryRouter>
    );
    await act(async () => navigate("/erudi/conversations/5", { replace: true, state: {} }));

    expect(deletes()).toHaveLength(0);
  });

  it("never deletes a regular conversation", async () => {
    renderAt("/erudi/conversations/5");

    await act(async () => navigate("/erudi/chat"));

    expect(deletes()).toHaveLength(0);
  });

  it("deletes once, however often the route changes afterwards", async () => {
    markEphemeral(5);
    renderAt("/erudi/conversations/5");

    await act(async () => navigate("/erudi/chat"));
    await act(async () => navigate("/erudi/conversations/5"));
    await act(async () => navigate("/erudi/chat"));

    expect(deletes()).toHaveLength(1);
  });

  it("aborts the answer streaming in the conversation before the delete", async () => {
    markEphemeral(5);
    const controller = new AbortController();
    trackTurn(5, controller);
    let abortedAtDelete = null;
    tracedFetchMock.mockImplementation(async () => {
      abortedAtDelete = controller.signal.aborted;
      return { ok: true, status: 200 };
    });
    renderAt("/erudi/conversations/5");

    await act(async () => navigate("/erudi/conversations/6"));

    expect(controller.signal.aborted).toBe(true);
    expect(abortedAtDelete).toBe(true);
  });
});
