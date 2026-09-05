// @vitest-environment jsdom
/**
 * The shared "here is what to send us" block.
 *
 * The assertions that matter are the query string of the GitHub link — a
 * prefill that silently stops working is invisible until a maintainer notices
 * empty fields — and the fact that the diagnostics text is never put in that
 * URL. GitHub answers 414 to a long URL, and a log tail is long.
 */
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

import ReportProblem, { CONTACT_PAGE_URL, buildIssueUrl } from "./ReportProblem";
import i18n from "../i18n";

const DIAGNOSTICS = "Erudi 1.0.0\nEngine: MLX_Engine\n[ERROR] boom";

let writeText;

beforeEach(() => {
  writeText = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(window.navigator, "clipboard", {
    value: { writeText },
    configurable: true,
  });
  window.open = vi.fn();
});

afterEach(async () => {
  cleanup();
  vi.restoreAllMocks();
  await i18n.changeLanguage("en");
});

describe("buildIssueUrl", () => {
  it("targets the bug report form of this repository", () => {
    const url = new URL(buildIssueUrl({}));
    expect(url.origin + url.pathname).toBe("https://github.com/erudi-app/erudi/issues/new");
    expect(url.searchParams.get("template")).toBe("bug_report.yml");
  });

  it("prefills the form fields by their template id", () => {
    const url = new URL(
      buildIssueUrl({
        version: "1.0.0",
        os: "macOS (Apple Silicon)",
        hardware: "Apple M3 Pro, arm64",
        model: "mlx-community/Qwen3-4B-4bit",
      })
    );
    expect(url.searchParams.get("version")).toBe("1.0.0");
    expect(url.searchParams.get("os")).toBe("macOS (Apple Silicon)");
    expect(url.searchParams.get("hardware")).toBe("Apple M3 Pro, arm64");
    expect(url.searchParams.get("model")).toBe("mlx-community/Qwen3-4B-4bit");
  });

  it("omits a field that has no value rather than sending an empty one", () => {
    const url = new URL(buildIssueUrl({ version: "1.0.0", model: null, hardware: "" }));
    expect(url.searchParams.has("model")).toBe(false);
    expect(url.searchParams.has("hardware")).toBe(false);
  });

  it("never carries the diagnostics text: a long URL is a 414", () => {
    const url = buildIssueUrl({ version: "1.0.0" }, DIAGNOSTICS);
    expect(url).not.toContain("boom");
    expect(url).not.toContain("logs=");
    expect(url.length).toBeLessThan(500);
  });
});

describe("ReportProblem", () => {
  it("shows the diagnostics text read-only", () => {
    render(<ReportProblem diagnostics={DIAGNOSTICS} />);
    const area = screen.getByLabelText("Diagnostics to copy");
    expect(area.value).toBe(DIAGNOSTICS);
    expect(area.readOnly).toBe(true);
  });

  it("copies the diagnostics and confirms it", async () => {
    render(<ReportProblem diagnostics={DIAGNOSTICS} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy" }));
    expect(writeText).toHaveBeenCalledWith(DIAGNOSTICS);
    await waitFor(() => expect(screen.getByText("Copied")).toBeTruthy());
  });

  it("copies the diagnostics before opening the prefilled issue form", async () => {
    render(
      <ReportProblem
        diagnostics={DIAGNOSTICS}
        prefill={{ version: "1.0.0", os: "Linux", hardware: "CPU only", model: null }}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: "Report on GitHub" }));
    await waitFor(() => expect(window.open).toHaveBeenCalled());
    // The clipboard is filled first: the form's Logs field is filled by paste.
    expect(writeText).toHaveBeenCalledWith(DIAGNOSTICS);
    const [href] = window.open.mock.calls[0];
    const url = new URL(href);
    expect(url.searchParams.get("version")).toBe("1.0.0");
    expect(url.searchParams.get("os")).toBe("Linux");
    expect(url.searchParams.get("hardware")).toBe("CPU only");
    expect(href).not.toContain("boom");
  });

  it("opens the issue form even when the clipboard refuses", async () => {
    writeText.mockRejectedValue(new Error("denied"));
    render(<ReportProblem diagnostics={DIAGNOSTICS} />);
    fireEvent.click(screen.getByRole("button", { name: "Report on GitHub" }));
    await waitFor(() => expect(window.open).toHaveBeenCalled());
  });

  it("offers the contact page for people without a GitHub account", () => {
    render(<ReportProblem diagnostics={DIAGNOSTICS} />);
    const link = screen.getByRole("link", { name: "Write to us on the contact page" });
    expect(link.getAttribute("href")).toBe(CONTACT_PAGE_URL);
    expect(link.getAttribute("target")).toBe("_blank");
    // The sentence is split around the link, so match the paragraph.
    expect(link.closest("p").textContent).toContain(
      "include everything above, plus any screenshots you have"
    );
  });

  it("translates every string it renders", async () => {
    await i18n.changeLanguage("fr");
    render(<ReportProblem diagnostics={DIAGNOSTICS} />);
    expect(screen.getByRole("button", { name: "Signaler sur GitHub" })).toBeTruthy();
    const link = screen.getByRole("link", { name: /page de contact/ });
    expect(link.closest("p").textContent).toContain("Pas de compte GitHub ?");
  });
});
