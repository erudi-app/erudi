// @vitest-environment jsdom
/**
 * The last stop for a render that throws.
 *
 * Without a boundary, React unmounts the whole tree and the window goes white
 * with nothing in either log file. The boundary has to do two things and
 * neither may fail: log the error the same way the global handlers do, and
 * put a recoverable screen on the glass.
 */
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

import AppErrorBoundary from "./AppErrorBoundary";
import { getSessionErrors, resetSessionErrors } from "../utils/errorCapture";
import i18n from "../i18n";

function Boom() {
  throw new Error("render exploded");
}

let consoleError;

beforeEach(() => {
  resetSessionErrors();
  window.logAPI = { send: vi.fn() };
  // React logs the caught error itself; that noise is not the test's subject.
  consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(async () => {
  cleanup();
  consoleError.mockRestore();
  delete window.logAPI;
  vi.restoreAllMocks();
  await i18n.changeLanguage("en");
});

describe("AppErrorBoundary", () => {
  it("renders its children when nothing throws", () => {
    render(
      <AppErrorBoundary>
        <p>all good</p>
      </AppErrorBoundary>
    );
    expect(screen.getByText("all good")).toBeTruthy();
  });

  it("shows a recoverable screen when a child throws", () => {
    render(
      <AppErrorBoundary>
        <Boom />
      </AppErrorBoundary>
    );
    expect(screen.getByText("This screen stopped working")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Reload the app" })).toBeTruthy();
  });

  it("records the error in the session buffer, the way the global handlers do", () => {
    render(
      <AppErrorBoundary>
        <Boom />
      </AppErrorBoundary>
    );
    const [entry] = getSessionErrors();
    expect(entry.origin).toBe("react.errorBoundary");
    expect(entry.message).toContain("render exploded");
    expect(window.logAPI.send).toHaveBeenCalled();
  });

  it("offers the shared report block with the error already in it", () => {
    render(
      <AppErrorBoundary>
        <Boom />
      </AppErrorBoundary>
    );
    expect(screen.getByRole("button", { name: "Report on GitHub" })).toBeTruthy();
    expect(screen.getByLabelText("Diagnostics to copy").value).toContain("render exploded");
  });

  it("reloads the window when asked", () => {
    const reload = vi.fn();
    Object.defineProperty(window, "location", {
      value: { ...window.location, reload },
      configurable: true,
      writable: true,
    });
    render(
      <AppErrorBoundary>
        <Boom />
      </AppErrorBoundary>
    );
    fireEvent.click(screen.getByRole("button", { name: "Reload the app" }));
    expect(reload).toHaveBeenCalled();
  });

  it("translates its screen", async () => {
    await i18n.changeLanguage("fr");
    render(
      <AppErrorBoundary>
        <Boom />
      </AppErrorBoundary>
    );
    expect(screen.getByText("Cet écran a cessé de fonctionner")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Recharger l'application" })).toBeTruthy();
  });
});
