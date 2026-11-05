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
    expect(toggle.textContent).toBe("Hide details");
    expect(screen.getByText(cause.text)).toBeTruthy();
    expect(screen.getByText("be-1f2e3d4c")).toBeTruthy();
    expect(screen.getByText("Request id")).toBeTruthy();

    fireEvent.click(toggle);
    expect(screen.queryByText(cause.text)).toBeNull();
  });

  it("copies the cause and the request id together", async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.assign(navigator, { clipboard: { writeText } });
    render(<ErrorDetails detail={cause} />);

    fireEvent.click(screen.getByRole("button", { name: "Details" }));
    fireEvent.click(screen.getByRole("button", { name: "Copy details" }));

    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith(`${cause.text}\nrequest_id: be-1f2e3d4c`)
    );
  });

  it("copies the cause alone when the request had no id", async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.assign(navigator, { clipboard: { writeText } });
    render(<ErrorDetails detail={{ ...cause, request_id: "-" }} />);

    fireEvent.click(screen.getByRole("button", { name: "Details" }));
    fireEvent.click(screen.getByRole("button", { name: "Copy details" }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(cause.text));
  });

  it("survives a clipboard that refuses", async () => {
    const writeText = vi.fn(() => Promise.reject(new Error("denied")));
    Object.assign(navigator, { clipboard: { writeText } });
    render(<ErrorDetails detail={cause} />);

    fireEvent.click(screen.getByRole("button", { name: "Details" }));
    fireEvent.click(screen.getByRole("button", { name: "Copy details" }));

    await waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(screen.getByRole("button", { name: "Copy details" })).toBeTruthy();
  });

  it("points aria-controls at the panel only while the panel exists", () => {
    render(<ErrorDetails detail={cause} />);
    const toggle = screen.getByRole("button", { name: "Details" });
    expect(toggle.getAttribute("aria-controls")).toBeNull();

    fireEvent.click(toggle);

    const panelId = toggle.getAttribute("aria-controls");
    expect(document.getElementById(panelId)).not.toBeNull();
  });

  it("omits the request id line when the request had none", () => {
    render(<ErrorDetails detail={{ ...cause, request_id: "-" }} />);

    fireEvent.click(screen.getByRole("button", { name: "Details" }));

    expect(screen.getByText(cause.text)).toBeTruthy();
    expect(screen.queryByText("Request id")).toBeNull();
  });
});
