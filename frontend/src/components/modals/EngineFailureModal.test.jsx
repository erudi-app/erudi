// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup, waitFor, act } from "@testing-library/react";
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
  // The shared report block reads the app's own version and platform from the
  // main process, because the backend may be the thing that just failed.
  window.diagnosticsAPI = {
    getAppInfo: vi.fn().mockResolvedValue({
      version: "1.0.0",
      platform: "win32",
      arch: "x64",
      appLogPath: "C:\\Temp\\erudi-backend.log",
    }),
  };
  window.open = vi.fn();
});

afterEach(() => {
  cleanup();
  delete window.backendAPI;
  delete window.diagnosticsAPI;
  delete navigator.clipboard;
  vi.restoreAllMocks();
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

  it("hands the failure to the app's one report block, trace included", () => {
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={() => {}} />);

    const area = screen.getByLabelText("Diagnostics to copy");
    expect(area.value).toContain(STARTUP_NOTICE.raw);
    // Plus what a maintainer cannot get from the trace alone.
    expect(area.value).toContain("CUDA_DRIVER_TOO_OLD");
    expect(area.value).toContain("NVIDIA GeForce GTX 1080");
    expect(area.value).toContain("6.1");
  });

  it("copies the report through the shared block", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={() => {}} />);

    fireEvent.click(screen.getByRole("button", { name: "Copy" }));

    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith(expect.stringContaining(STARTUP_NOTICE.raw))
    );
  });

  it("prefills the issue form with the card that failed", async () => {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true,
    });
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={() => {}} />);
    // The app info arrives from the main process a microtask later; the
    // prefill is read at click time, so let it land first.
    await waitFor(() => expect(window.diagnosticsAPI.getAppInfo).toHaveBeenCalled());
    await act(async () => {});

    fireEvent.click(screen.getByRole("button", { name: "Report on GitHub" }));

    await waitFor(() => expect(window.open).toHaveBeenCalled());
    const url = new URL(window.open.mock.calls[0][0]);
    expect(url.searchParams.get("template")).toBe("bug_report.yml");
    expect(url.searchParams.get("version")).toBe("1.0.0");
    expect(url.searchParams.get("os")).toBe("Windows 10 / 11");
    expect(url.searchParams.get("hardware")).toBe("NVIDIA GeForce GTX 1080, compute 6.1");
  });

  it("offers the contact page as the route for reporters without GitHub", () => {
    render(<EngineFailureModal notice={STARTUP_NOTICE} onDismiss={() => {}} />);

    const site = screen.getByRole("link", { name: /contact page/i });
    expect(site.getAttribute("href")).toBe("https://erudi.app/contact");
    expect(site.getAttribute("target")).toBe("_blank");
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
    expect(screen.getByLabelText("Diagnostics to copy").value).toContain(
      "CUDA error: out of memory"
    );
  });
});
