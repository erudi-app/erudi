// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act } from "@testing-library/react";

import CommandNotice from "./CommandNotice.jsx";

// The one-line outcome of a command above the composer (#585): an
// informative notice goes on its own after a few seconds; an error stays
// until it is closed.

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("CommandNotice", () => {
  it("renders nothing without a notice", () => {
    const { container } = render(<CommandNotice notice={null} onDismiss={vi.fn()} />);
    expect(container.innerHTML).toBe("");
  });

  it("lets an informative notice go after a few seconds", () => {
    vi.useFakeTimers();
    const onDismiss = vi.fn();
    render(
      <CommandNotice
        notice={{ tone: "info", text: "Nothing to compact yet." }}
        onDismiss={onDismiss}
      />
    );
    expect(screen.getByRole("status").textContent).toBe("Nothing to compact yet.");
    act(() => vi.advanceTimersByTime(5999));
    expect(onDismiss).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1));
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("keeps an error until it is closed", () => {
    vi.useFakeTimers();
    const onDismiss = vi.fn();
    render(
      <CommandNotice notice={{ tone: "error", text: "The model is busy." }} onDismiss={onDismiss} />
    );
    expect(screen.getByRole("alert").textContent).toBe("The model is busy.");
    act(() => vi.advanceTimersByTime(60_000));
    expect(onDismiss).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });
});
