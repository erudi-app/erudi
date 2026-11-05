// @vitest-environment jsdom
/**
 * The "Updated to X.Y.Z" dialog (#572). It renders a decision taken by
 * `decideUpdateNotice`: the notes as plain text lines when they are this
 * version's, the release page link always, and the opt-in explanation on the
 * first launch of this version over an older one.
 */
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

import UpdateNoticeDialog from "./UpdateNoticeDialog";

afterEach(cleanup);

const RELEASE = "https://github.com/erudi-app/erudi/releases/tag/v1.1.5";

describe("UpdateNoticeDialog", () => {
  it("renders nothing without a notice", () => {
    const { container } = render(<UpdateNoticeDialog notice={null} onDismiss={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it("names the version and lists the notes as text", () => {
    render(
      <UpdateNoticeDialog
        notice={{
          version: "1.1.5",
          lines: ["What's Changed", "• <b>faster</b> loads"],
          explainOptIns: false,
        }}
        onDismiss={vi.fn()}
      />
    );

    expect(screen.getByRole("dialog", { name: "Updated to 1.1.5" })).toBeDefined();
    expect(screen.getByText("What's Changed")).toBeDefined();
    // Text, never markup: the angle brackets are shown, nothing is parsed.
    expect(screen.getByText("• <b>faster</b> loads")).toBeDefined();
    expect(document.querySelector("b")).toBeNull();
    expect(document.querySelector("img")).toBeNull();
  });

  it("always links the release page, opened in the browser only on click", () => {
    render(
      <UpdateNoticeDialog
        notice={{ version: "1.1.5", lines: ["One"], explainOptIns: false }}
        onDismiss={vi.fn()}
      />
    );

    const link = screen.getByRole("link", { name: "Read the release notes" });
    expect(link.getAttribute("href")).toBe(RELEASE);
    expect(link.getAttribute("target")).toBe("_blank");
    expect(document.querySelectorAll("a")).toHaveLength(1);
  });

  it("says the notes are on the release page when there are none", () => {
    render(
      <UpdateNoticeDialog
        notice={{ version: "1.1.5", lines: null, explainOptIns: false }}
        onDismiss={vi.fn()}
      />
    );

    expect(screen.getByText(/release page/)).toBeDefined();
    expect(screen.getByRole("link", { name: "Read the release notes" })).toBeDefined();
  });

  it("explains that downloading and installing are now two choices, both off", () => {
    render(
      <UpdateNoticeDialog
        notice={{ version: "1.1.4", lines: null, explainOptIns: true }}
        onDismiss={vi.fn()}
      />
    );

    const text = document.body.textContent;
    expect(text).toMatch(/Downloading and installing updates are now two separate choices/);
    expect(text).toMatch(/both off/);
    // True whatever Check for updates was set to (P3-8): no promise of a banner.
    expect(text).not.toMatch(/banner/i);
    expect(text).toMatch(/Check for updates keeps the value it had/);
  });

  it("makes no banner promise in any language", async () => {
    for (const lang of ["fr", "es", "zh"]) {
      const common = (await import(`../../locales/${lang}/common.json`)).default;
      expect(common.updateNotice.optIns, lang).not.toMatch(/bandeau|aviso|横幅/);
    }
  });

  it("does not explain the opt-ins on a later update", () => {
    render(
      <UpdateNoticeDialog
        notice={{ version: "1.1.5", lines: null, explainOptIns: false }}
        onDismiss={vi.fn()}
      />
    );

    expect(document.body.textContent).not.toMatch(/separate choices/);
  });

  it("is dismissed by its button and by Escape, with the focus on the button at open", () => {
    const onDismiss = vi.fn();
    render(
      <UpdateNoticeDialog
        notice={{ version: "1.1.5", lines: null, explainOptIns: false }}
        onDismiss={onDismiss}
      />
    );

    const close = screen.getByRole("button", { name: "Close" });
    expect(document.activeElement).toBe(close);

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onDismiss).toHaveBeenCalledTimes(1);

    fireEvent.click(close);
    expect(onDismiss).toHaveBeenCalledTimes(2);
  });
});
