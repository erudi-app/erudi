// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, waitFor } from "@testing-library/react";

// The trash buttons of the Local Models list carry an aria-label (#766).

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn(async () => []) },
  apiClient: { get: vi.fn(async () => []) },
  tracedFetch: tracedFetchMock,
}));
vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: vi.fn(), completionCount: 0 }),
}));

import ModelCollapsibleSection from "./ModelCollapsibleSection.jsx";

afterEach(cleanup);

describe("Local Models list delete button", () => {
  it("has an aria-label", async () => {
    tracedFetchMock.mockImplementation(async (url) => ({
      ok: true,
      status: 200,
      json: async () =>
        String(url).endsWith("/llms/local") ? [{ id: 319, name: "Qwen3 4B", link: "/m/319" }] : [],
    }));
    render(<ModelCollapsibleSection kind="local" />);
    await waitFor(() => expect(screen.getByText("Qwen3 4B")).toBeDefined(), { timeout: 4000 });
    expect(screen.getByTitle("Delete model").getAttribute("aria-label")).toBe("Delete model");
  });
});
