// @vitest-environment jsdom
// The size a Hugging Face search hit shows (#761): the measured download size
// when the backend read it, the "~" estimate otherwise. The real card renders
// here, unlike HuggingFaceSearchPanel.test.jsx, which stubs it.
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

vi.mock("../services/api/client", () => {
  const apiClient = { get: vi.fn() };
  return { apiClient, default: apiClient };
});
vi.mock("../utils/logger", () => ({
  createLogger: () => ({ error: vi.fn(), warn: vi.fn(), info: vi.fn(), debug: vi.fn() }),
}));

import apiClient from "../services/api/client";
import HuggingFaceSearchPanel from "./HuggingFaceSearchPanel";

const hit = (overrides = {}) => ({
  name: "Qwen2.5 VL 3B Instruct 4bit",
  link: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
  category: "vision",
  param_size: 4,
  quantized: true,
  gated: false,
  downloads: 100,
  likes: 10,
  pipeline_tag: "image-text-to-text",
  ...overrides,
});

async function searchFor(term) {
  render(<HuggingFaceSearchPanel range={{ min: 2, max: 8 }} />);
  fireEvent.change(screen.getByPlaceholderText(/Try "qwen coder"/), {
    target: { value: term },
  });
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  await waitFor(() =>
    expect(screen.queryByText(/searching hugging face for/)).not.toBeInTheDocument()
  );
}

let onLineSpy;

beforeEach(() => {
  vi.clearAllMocks();
  onLineSpy = vi.spyOn(window.navigator, "onLine", "get").mockReturnValue(true);
});

afterEach(() => {
  cleanup();
  onLineSpy.mockRestore();
});

describe("HuggingFaceSearchPanel hit size", () => {
  it("shows the measured size, without the ~, when the hit carries one", async () => {
    apiClient.get.mockResolvedValueOnce([hit({ artifact_size_bytes: 3_089_713_215 })]);
    await searchFor("qwen vl");
    expect(screen.getByText("3.1 GB")).toBeInTheDocument();
    expect(screen.queryByText("~2.4 GB")).not.toBeInTheDocument();
  });

  it("keeps the ~ estimate when the size could not be read", async () => {
    apiClient.get.mockResolvedValueOnce([hit({ artifact_size_bytes: null })]);
    await searchFor("qwen vl");
    expect(screen.getByText("~2.4 GB")).toBeInTheDocument();
  });

  it("keeps the ~ estimate when the hit has no size field at all", async () => {
    apiClient.get.mockResolvedValueOnce([hit()]);
    await searchFor("qwen vl");
    expect(screen.getByText("~2.4 GB")).toBeInTheDocument();
  });
});
