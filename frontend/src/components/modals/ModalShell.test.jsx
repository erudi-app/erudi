// @vitest-environment jsdom
import React, { useState } from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// The shell every command popup shares (#585): it takes the focus when it
// opens, keeps Tab inside, closes on Escape, and hands the focus back.

import ModalShell from "./ModalShell.jsx";
import HelpTip from "../HelpTip";

afterEach(cleanup);

function Harness({ onClose = () => {} }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button onClick={() => setOpen(true)}>opener</button>
      <ModalShell
        isOpen={open}
        onClose={() => {
          onClose();
          setOpen(false);
        }}
        title="Popup title"
        subtitle="Popup subtitle"
      >
        <button>first</button>
        <input aria-label="middle" />
        <button>last</button>
      </ModalShell>
    </div>
  );
}

const open = () => {
  const opener = screen.getByText("opener");
  opener.focus();
  fireEvent.click(opener);
  return opener;
};

describe("ModalShell", () => {
  it("is a labelled modal dialog carrying its subtitle", () => {
    render(<Harness />);
    open();
    const dialog = screen.getByRole("dialog");
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(screen.getByText("Popup title").id).toBe(dialog.getAttribute("aria-labelledby"));
    expect(screen.getByText("Popup subtitle").id).toBe(dialog.getAttribute("aria-describedby"));
  });

  it("moves the focus into the dialog when it opens", async () => {
    render(<Harness />);
    open();
    await waitFor(() =>
      expect(screen.getByRole("dialog").contains(document.activeElement)).toBe(true)
    );
  });

  it("focuses the first field, not a ? help button (its tooltip would cover the field)", async () => {
    render(
      <ModalShell isOpen onClose={() => {}} title="T">
        <HelpTip label="Help: Field" text="Some help" />
        <input aria-label="field" />
      </ModalShell>
    );
    await waitFor(() => expect(document.activeElement).toBe(screen.getByLabelText("field")));
  });

  it("cycles Tab and Shift+Tab inside the dialog", async () => {
    render(<Harness />);
    open();
    const dialog = await screen.findByRole("dialog");
    const close = screen.getByRole("button", { name: "Close" });
    const last = screen.getByText("last");

    last.focus();
    fireEvent.keyDown(dialog, { key: "Tab" });
    expect(document.activeElement).toBe(close);

    fireEvent.keyDown(dialog, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(last);
  });

  it("closes on Escape and gives the focus back to what had it", async () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose} />);
    const opener = open();
    const dialog = await screen.findByRole("dialog");

    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(document.activeElement).toBe(opener);
  });

  it("closes from its close button", async () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose} />);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

// A click on a heading, a label or the padding leaves nothing focused inside
// the popup: the keys must still reach it.
describe("ModalShell after a click on a non-focusable spot", () => {
  it("still closes on Escape sent from the page body", async () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose} />);
    open();
    await screen.findByRole("dialog");
    document.activeElement.blur();
    expect(document.activeElement).toBe(document.body);
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("brings Tab and Shift+Tab back into the dialog", async () => {
    render(<Harness />);
    open();
    await screen.findByRole("dialog");
    document.activeElement.blur();
    fireEvent.keyDown(document.body, { key: "Tab" });
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Close" }));
    document.activeElement.blur();
    fireEvent.keyDown(document.body, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(screen.getByText("last"));
  });

  it("ignores an Escape that only cancels an input method's composition", async () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose} />);
    open();
    const dialog = await screen.findByRole("dialog");
    const field = screen.getByLabelText("middle");
    fireEvent.keyDown(field, { key: "Escape", isComposing: true });
    fireEvent.keyDown(field, { key: "Escape", keyCode: 229 });
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("lets a click on the panel focus the panel itself", async () => {
    render(<Harness />);
    open();
    expect((await screen.findByRole("dialog")).tabIndex).toBe(-1);
  });

  it("stops listening once closed", async () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose} />);
    open();
    fireEvent.keyDown(await screen.findByRole("dialog"), { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
