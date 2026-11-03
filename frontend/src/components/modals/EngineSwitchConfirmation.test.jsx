// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { EngineSwitchConfirmationContent } from "./EngineSwitchConfirmation";

afterEach(cleanup);

describe("EngineSwitchConfirmationContent", () => {
  it("answers once: a double click on the switch confirms a single time", () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    render(<EngineSwitchConfirmationContent onConfirm={onConfirm} onCancel={onCancel} />);

    const confirm = screen.getByRole("button", { name: "Cancel the download and switch" });
    fireEvent.click(confirm);
    fireEvent.click(confirm);
    fireEvent.click(screen.getByRole("button", { name: "Keep downloading" }));

    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
  });
});
