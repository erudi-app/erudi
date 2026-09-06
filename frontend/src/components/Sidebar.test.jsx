// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

const mockDownloadModal = { isDownloading: false };
vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => mockDownloadModal,
}));

// The badge is a rendering concern of Sidebar; what feeds its number
// (polling, the predicate, freshness) is unit-tested on its own in
// shared/hooks/useBugCounter.test.js and utils/bugCounter.test.js.
const mockBugCounter = { count: 0, label: "" };
vi.mock("../shared/hooks/useBugCounter", () => ({
  default: () => mockBugCounter,
}));

import Sidebar from "./Sidebar";

function renderAt(path, props = {}) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Sidebar {...props} />
    </MemoryRouter>
  );
}

afterEach(() => {
  cleanup();
  mockDownloadModal.isDownloading = false;
  mockBugCounter.count = 0;
  mockBugCounter.label = "";
  vi.restoreAllMocks();
});

describe("Sidebar", () => {
  it("renders the four navigation links with their routes", () => {
    renderAt("/erudi/models");
    expect(screen.getByLabelText("Models").getAttribute("href")).toBe("/erudi/models");
    expect(screen.getByLabelText("Chat").getAttribute("href")).toBe("/erudi/chat");
    expect(screen.getByLabelText("Arena").getAttribute("href")).toBe("/erudi/arena");
    expect(screen.getByLabelText("Knowledge Base").getAttribute("href")).toBe(
      "/erudi/attach_knowledge_base"
    );
  });

  it("highlights the entry matching the current route", () => {
    renderAt("/erudi/arena");
    expect(screen.getByLabelText("Arena").className).toContain("border-green-500");
    expect(screen.getByLabelText("Models").className).toContain("border-transparent");
  });

  it("treats conversation routes as the chat section", () => {
    renderAt("/erudi/conversations/42");
    expect(screen.getByLabelText("Chat").className).toContain("border-green-500");
  });

  it("disables pointer events when disabled", () => {
    const { container } = renderAt("/erudi/models", { disabled: true });
    expect(container.firstChild.className).toContain("pointer-events-none");
  });

  it("replaces the chat link with a toggle button when showCollapsible is set", () => {
    const onToggleSidebar = vi.fn();
    renderAt("/erudi/chat", { showCollapsible: true, onToggleSidebar });
    const toggle = screen.getByLabelText("Toggle chat sidebar");
    fireEvent.click(toggle);
    expect(onToggleSidebar).toHaveBeenCalledTimes(1);
    expect(screen.queryByLabelText("Chat")).toBeNull();
  });

  it("swaps the chat icon to a panel icon on hover, matching the collapsed state", () => {
    renderAt("/erudi/chat", { showCollapsible: true, collapsed: false });
    const toggle = screen.getByLabelText("Toggle chat sidebar");
    expect(document.querySelector(".lucide-panel-left-close")).toBeNull();
    fireEvent.mouseEnter(toggle);
    expect(document.querySelector(".lucide-panel-left-close")).not.toBeNull();
    fireEvent.mouseLeave(toggle);
    expect(document.querySelector(".lucide-panel-left-close")).toBeNull();
  });

  it("shows the open-panel icon on hover when the chat sidebar is collapsed", () => {
    renderAt("/erudi/chat", { showCollapsible: true, collapsed: true });
    fireEvent.mouseEnter(screen.getByLabelText("Toggle chat sidebar"));
    expect(document.querySelector(".lucide-panel-left-open")).not.toBeNull();
  });

  it("replaces the models link with a brain toggle when showBrainCollapsible is set", () => {
    const onToggleBrainSidebar = vi.fn();
    renderAt("/erudi/models", { showBrainCollapsible: true, onToggleBrainSidebar });
    const toggle = screen.getByLabelText("Toggle models sidebar");
    fireEvent.click(toggle);
    expect(onToggleBrainSidebar).toHaveBeenCalledTimes(1);
    expect(screen.queryByLabelText("Models")).toBeNull();
  });

  it("swaps the brain icon on hover based on the collapsed state", () => {
    renderAt("/erudi/models", { showBrainCollapsible: true, brainCollapsed: true });
    const toggle = screen.getByLabelText("Toggle models sidebar");
    fireEvent.mouseEnter(toggle);
    expect(document.querySelector(".lucide-panel-left-open")).not.toBeNull();
    fireEvent.mouseLeave(toggle);
    expect(document.querySelector(".lucide-brain")).not.toBeNull();
  });

  it("sends the bug report button to the Diagnostics page, not straight to a web page", () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => {});
    renderAt("/erudi/models");
    const button = screen.getByLabelText("Report a bug");
    // In-app first: the page shows the user what to send before any link out.
    expect(button.getAttribute("href")).toBe("/erudi/diagnostics");
    fireEvent.click(button);
    expect(open).not.toHaveBeenCalled();
  });

  it("highlights the bug button on the diagnostics route, like any other destination", () => {
    renderAt("/erudi/diagnostics");
    const button = screen.getByLabelText("Report a bug");
    expect(button.className).toContain("border-green-500");
    expect(button.querySelector("svg").getAttribute("class")).toContain("text-green-400");
    expect(screen.getByLabelText("Settings").className).toContain("border-transparent");
  });

  it("hovers the bug button green rather than red: it is a destination, not an alarm", () => {
    renderAt("/erudi/models");
    const icon = screen.getByLabelText("Report a bug").querySelector("svg");
    expect(icon.getAttribute("class")).toContain("hover:text-green-400");
    expect(icon.getAttribute("class")).not.toContain("red");
  });

  it("hides the bug report button during a download", () => {
    mockDownloadModal.isDownloading = true;
    renderAt("/erudi/models");
    expect(screen.queryByLabelText("Report a bug")).toBeNull();
  });

  it("renders the settings gear at the bottom, linking to the settings page", () => {
    renderAt("/erudi/models");
    expect(screen.getByLabelText("Settings").getAttribute("href")).toBe("/erudi/settings");
  });

  it("highlights the settings gear on the settings route", () => {
    renderAt("/erudi/settings");
    expect(screen.getByLabelText("Settings").className).toContain("border-green-500");
    expect(screen.getByLabelText("Models").className).toContain("border-transparent");
  });

  it("keeps the settings gear visible during a download", () => {
    mockDownloadModal.isDownloading = true;
    renderAt("/erudi/models");
    expect(screen.getByLabelText("Settings")).toBeTruthy();
  });

  describe("bug counter badge (#485)", () => {
    it("shows no badge and the plain label when the count is 0", () => {
      mockBugCounter.count = 0;
      mockBugCounter.label = "";
      renderAt("/erudi/models");
      const button = screen.getByLabelText("Report a bug");
      expect(button.querySelector('[data-testid="bug-badge"]')).toBeNull();
    });

    it("shows the exact count on the badge", () => {
      mockBugCounter.count = 3;
      mockBugCounter.label = "3";
      renderAt("/erudi/models");
      const badge = screen
        .getByLabelText("Report a bug (3 new errors)")
        .querySelector('[data-testid="bug-badge"]');
      expect(badge).not.toBeNull();
      expect(badge.textContent).toBe("3");
    });

    it("caps the badge text at 9+ beyond nine", () => {
      mockBugCounter.count = 12;
      mockBugCounter.label = "9+";
      renderAt("/erudi/models");
      const badge = screen
        .getByLabelText("Report a bug (12 new errors)")
        .querySelector('[data-testid="bug-badge"]');
      expect(badge.textContent).toBe("9+");
    });

    it("uses the singular aria-label for exactly one new error", () => {
      mockBugCounter.count = 1;
      mockBugCounter.label = "1";
      renderAt("/erudi/models");
      expect(screen.getByLabelText("Report a bug (1 new error)")).toBeTruthy();
    });

    it("shows the badge on the diagnostics route too, alongside the active highlight", () => {
      mockBugCounter.count = 2;
      mockBugCounter.label = "2";
      renderAt("/erudi/diagnostics");
      const button = screen.getByLabelText("Report a bug (2 new errors)");
      expect(button.className).toContain("border-green-500");
      expect(button.querySelector('[data-testid="bug-badge"]').textContent).toBe("2");
    });
  });
});
