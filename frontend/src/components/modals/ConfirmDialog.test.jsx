// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

import ConfirmDialog from "./ConfirmDialog.jsx";

// A generic confirmation in the command popups' shell (#585): it explains,
// then the user confirms or backs out; backing out is the safe default.

let onConfirm;
let onClose;

const renderDialog = (props = {}) =>
  render(
    <ConfirmDialog
      isOpen
      title="Do the thing?"
      message="The thing cannot be undone."
      confirmLabel="Do it"
      onConfirm={onConfirm}
      onClose={onClose}
      {...props}
    />
  );

beforeEach(() => {
  onConfirm = vi.fn();
  onClose = vi.fn();
});

afterEach(cleanup);

describe("ConfirmDialog", () => {
  it("explains what confirming does", () => {
    renderDialog();
    const dialog = screen.getByRole("dialog", { name: "Do the thing?" });
    expect(dialog.getAttribute("aria-describedby")).toBeTruthy();
    expect(document.getElementById(dialog.getAttribute("aria-describedby")).textContent).toBe(
      "The thing cannot be undone."
    );
  });

  it("confirms once from its confirm button", () => {
    renderDialog();
    fireEvent.click(screen.getByRole("button", { name: "Do it" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onClose).not.toHaveBeenCalled();
  });

  it("confirms once per opening, whatever the clicks during its closing", () => {
    const { rerender } = renderDialog();
    const confirm = screen.getByRole("button", { name: "Do it" });
    fireEvent.click(confirm);
    fireEvent.click(confirm);
    rerender(
      <ConfirmDialog
        isOpen={false}
        title="Do the thing?"
        message="The thing cannot be undone."
        confirmLabel="Do it"
        onConfirm={onConfirm}
        onClose={onClose}
      />
    );
    // The closing dialog may still be on screen for its exit animation.
    if (confirm.isConnected) fireEvent.click(confirm);
    expect(onConfirm).toHaveBeenCalledTimes(1);

    rerender(
      <ConfirmDialog
        isOpen
        title="Do the thing?"
        message="The thing cannot be undone."
        confirmLabel="Do it"
        onConfirm={onConfirm}
        onClose={onClose}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: "Do it" }));
    expect(onConfirm).toHaveBeenCalledTimes(2);
  });

  it("puts the focus on Cancel, so Enter right away backs out", () => {
    renderDialog();
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Cancel" }));
  });

  it.each([
    ["Cancel", () => fireEvent.click(screen.getByRole("button", { name: "Cancel" }))],
    ["Escape", () => fireEvent.keyDown(document, { key: "Escape" })],
    ["the close button", () => fireEvent.click(screen.getByRole("button", { name: "Close" }))],
  ])("closes without confirming on %s", (_label, dismiss) => {
    renderDialog();
    dismiss();
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("renders nothing while closed", () => {
    renderDialog({ isOpen: false });
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
