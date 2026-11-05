// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

import ErrorDetails from "./ErrorDetails";

// #615 - the discreet "Details" control under a generic failure: the cleaned
// cause, verbatim, and the request id, copyable.

const cause = {
  t: "error_detail",
  text: "Message content too long (max 32K chars)",
  request_id: "be-1f2e3d4c",
};

afterEach(() => {
  cleanup();
});

describe("ErrorDetails", () => {
  it("is collapsed until the control is used", () => {
    render(<ErrorDetails detail={cause} />);

    const toggle = screen.getByRole("button", { name: "Details" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText(cause.text)).toBeNull();

    fireEvent.click(toggle);

    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText(cause.text)).toBeTruthy();
    expect(screen.getByText("be-1f2e3d4c")).toBeTruthy();
    expect(screen.getByText("Request id")).toBeTruthy();

    fireEvent.click(toggle);
    expect(screen.queryByText(cause.text)).toBeNull();
  });

  it("copies the request id", async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.assign(navigator, { clipboard: { writeText } });
    render(<ErrorDetails detail={cause} />);

    fireEvent.click(screen.getByRole("button", { name: "Details" }));
    fireEvent.click(screen.getByRole("button", { name: "Copy request id" }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith("be-1f2e3d4c"));
  });

  it("omits the request id line when the request had none", () => {
    render(<ErrorDetails detail={{ ...cause, request_id: "-" }} />);

    fireEvent.click(screen.getByRole("button", { name: "Details" }));

    expect(screen.getByText(cause.text)).toBeTruthy();
    expect(screen.queryByText("Request id")).toBeNull();
  });
});
