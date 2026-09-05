// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor } from "@testing-library/react";
import EngineFailureModal from "./EngineFailureModal.jsx";
import { apiClient } from "../../services/api/client";

vi.mock("../../services/api/client", () => ({
  apiClient: { put: vi.fn() },
}));

const STARTUP_NOTICE = {
  event: "engine_notice",
  code: "CUDA_DRIVER_TOO_OLD",
  gpu_name: "NVIDIA GeForce GTX 1080",
  compute_capability: "6.1",
  driver_cuda_version: "12.1",
  required_cuda_version: "12.8",
  raw: "GTX 1080 needs a driver providing CUDA 12.8 or newer; this system reports CUDA 12.1.",
};

beforeEach(() => {
  apiClient.put.mockReset();
  apiClient.put.mockResolvedValue({ inference_backend: "cpu" });
  window.backendAPI = { restartBackend: vi.fn().mockResolvedValue(undefined) };
});

afterEach(() => {
  cleanup();
  delete window.backendAPI;
  delete navigator.clipboard;
});

describe("EngineFailureModal", () => {
  it("renders copy specific to the code, with the versions that matter", () => {
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={() => {}} />);

    expect(screen.getByText(/driver is too old/i)).toBeTruthy();
    // The required version is named, as is the one installed.
    expect(
      screen.getByText(
        /NVIDIA GeForce GTX 1080 needs an NVIDIA driver that provides CUDA 12\.8 or newer\. This computer reports CUDA 12\.1\./
      )
    ).toBeTruthy();
    // And the readings appear on their own, as detected facts.
    expect(screen.getByText(/what erudi detected/i)).toBeTruthy();
    expect(screen.getByText("6.1")).toBeTruthy();
    // Updating the driver is stated as the fix.
    expect(screen.getByText(/Updating the NVIDIA driver is the fix/i)).toBeTruthy();
  });

  it("falls back to generic copy for an unknown code", () => {
    render(<EngineFailureModal notice={{ code: "SOMETHING_NEW" }} onDismiss={() => {}} />);

    expect(screen.getByText(/GPU mode stopped working/i)).toBeTruthy();
  });

  it("shows the trace and copies it to the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={() => {}} />);

    expect(screen.getByText(STARTUP_NOTICE.raw)).toBeTruthy();
    fireEvent.click(screen.getByLabelText(/copy the technical details/i));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(STARTUP_NOTICE.raw));
  });

  it("offers both report destinations as external links", () => {
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={() => {}} />);

    const github = screen.getByRole("link", { name: /report on github/i });
    expect(github.getAttribute("href")).toBe("https://github.com/erudi-app/erudi/issues/new");
    expect(github.getAttribute("target")).toBe("_blank");

    const site = screen.getByRole("link", { name: /erudi website/i });
    expect(site.getAttribute("href")).toBe("https://erudi-app.github.io/erudi/");
  });

  it("links to the processor build on the releases page", () => {
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={() => {}} />);

    const installer = screen.getByRole("link", { name: /download the processor version/i });
    expect(installer.getAttribute("href")).toBe(
      "https://github.com/erudi-app/erudi/releases/latest"
    );
  });

  it("persists the CPU choice and restarts the backend", async () => {
    const onDismiss = vi.fn();
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={onDismiss} />);

    fireEvent.click(screen.getByRole("button", { name: /switch to processor mode/i }));

    await waitFor(() =>
      expect(apiClient.put).toHaveBeenCalledWith("/user_settings/", {
        inference_backend: "cpu",
      })
    );
    await waitFor(() => expect(window.backendAPI.restartBackend).toHaveBeenCalled());
    await waitFor(() => expect(onDismiss).toHaveBeenCalled());
  });

  it("does not restart the backend when the setting could not be saved", async () => {
    apiClient.put.mockRejectedValue(new Error("backend is down"));
    const onDismiss = vi.fn();
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={onDismiss} />);

    fireEvent.click(screen.getByRole("button", { name: /switch to processor mode/i }));

    await waitFor(() => expect(screen.getByText(/did not go through/i)).toBeTruthy());
    expect(window.backendAPI.restartBackend).not.toHaveBeenCalled();
    expect(onDismiss).not.toHaveBeenCalled();
  });

  it("dismisses without persisting anything on Not now", () => {
    const onDismiss = vi.fn();
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={onDismiss} />);

    fireEvent.click(screen.getByRole("button", { name: /not now/i }));

    expect(onDismiss).toHaveBeenCalled();
    expect(apiClient.put).not.toHaveBeenCalled();
  });

  it("renders nothing without a notice", () => {
    const { container } = render(<EngineFailureModal notice={null} onDismiss={() => {}} />);
    expect(container.textContent).toBe("");
  });

  it("hides the detected-hardware block when the readings are unknown", () => {
    // The runtime path carries a code and a trace, never the NVML readings.
    render(
      <EngineFailureModal
        notice={{ code: "CUDA_OUT_OF_MEMORY", raw: "CUDA error: out of memory" }}
        onDismiss={() => {}}
      />
    );

    expect(screen.queryByText(/what erudi detected/i)).toBeNull();
    expect(screen.getByText("CUDA error: out of memory")).toBeTruthy();
  });
});
