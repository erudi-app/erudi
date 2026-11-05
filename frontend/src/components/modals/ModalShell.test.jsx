// @vitest-environment jsdom
import React, { useState } from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// The shell every command popup shares (#585): it takes the focus when it
// opens, keeps Tab inside, closes on Escape, and hands the focus back.

import ModalShell from "./ModalShell.jsx";

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
