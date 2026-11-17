// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import ExploreModelCard from "./ExploreModelCard";

const baseModel = {
  id: 1,
  name: "Qwen2.5 7B Instruct",
  category: "code",
  param_size: 7,
  quantized: true,
  downloads: "123456",
  likes: "2000000",
};

afterEach(cleanup);

describe("ExploreModelCard", () => {
  it("shows the name, category label and formatted metrics", () => {
    render(<ExploreModelCard model={baseModel} range={{ min: 2, max: 8 }} />);
    expect(screen.getByText("Qwen2.5 7B Instruct")).toBeTruthy();
    expect(screen.getByText("Code")).toBeTruthy();
    expect(screen.getByText("7B")).toBeTruthy();
    expect(screen.getByText("123k")).toBeTruthy(); // downloads
    expect(screen.getByText("2M")).toBeTruthy(); // likes
  });

  it("formats sub-billion sizes in millions and >=10B without decimals", () => {
    render(<ExploreModelCard model={{ ...baseModel, param_size: 0.27 }} />);
    expect(screen.getByText("270M")).toBeTruthy();
    render(<ExploreModelCard model={{ ...baseModel, param_size: 14 }} />);
    expect(screen.getByText("14B")).toBeTruthy();
  });

  it("says 'Size unknown' instead of implying a value", () => {
    render(<ExploreModelCard model={{ ...baseModel, param_size: undefined }} />);
    expect(screen.getByText("Size unknown")).toBeTruthy();
  });

  it("shows the measured download size on the card face, like the info modal", () => {
    // Qwen2.5 VL 3B: the estimate said "~2.3 GB" while the modal and the
    // installed card said 3.1 GB (#397). One number everywhere.
    render(
      <ExploreModelCard
        model={{ ...baseModel, param_size: 3.75, artifact_size_bytes: 3_100_000_000 }}
        range={{ min: 2, max: 8 }}
      />
    );
    expect(screen.getByText("3.1 GB")).toBeTruthy();
    expect(screen.queryByText("~2.3 GB")).toBeNull();
  });

  it("formats ten-million-plus counts without a decimal", () => {
    render(<ExploreModelCard model={{ ...baseModel, downloads: "12345678", likes: "512" }} />);
    expect(screen.getByText("12M")).toBeTruthy();
    expect(screen.getByText("512")).toBeTruthy();
  });

  it("marks gated models", () => {
    render(<ExploreModelCard model={{ ...baseModel, gated: true }} />);
    expect(screen.getByText("gated")).toBeTruthy();
  });

  it("badges team-tested models", () => {
    render(<ExploreModelCard model={{ ...baseModel, name: "Qwen3 0.6B" }} />);
    expect(screen.getByTitle(/Tested by the Erudi team/)).toBeTruthy();
  });

  it("shows the vision icon for multimodal models", () => {
    render(<ExploreModelCard model={{ ...baseModel, category: "vision" }} />);
    expect(screen.getByTitle("Supports image input (vision)")).toBeTruthy();
    expect(screen.getByText("Vision & Multimodal")).toBeTruthy();
  });

  it("fires onInfo and onDownload with the model", () => {
    const onInfo = vi.fn();
    const onDownload = vi.fn();
    render(<ExploreModelCard model={baseModel} onInfo={onInfo} onDownload={onDownload} />);
    fireEvent.click(screen.getByText("Details"));
    expect(onInfo).toHaveBeenCalledWith(baseModel);
    fireEvent.click(screen.getByText("Download"));
    expect(onDownload).toHaveBeenCalledWith(baseModel);
  });

  it("disables download for models that cannot run on this hardware", () => {
    const onDownload = vi.fn();
    render(<ExploreModelCard model={{ ...baseModel, runnable: false }} onDownload={onDownload} />);
    expect(screen.getByText("Not supported on your hardware")).toBeTruthy();
    const button = screen.getByText("Download");
    expect(button.disabled).toBe(true);
    fireEvent.click(button);
    expect(onDownload).not.toHaveBeenCalled();
  });
});

describe("ExploreModelCard installed state (#348)", () => {
  it("reads Installed and refuses to re-download a model already on disk", () => {
    const onDownload = vi.fn();
    render(<ExploreModelCard model={baseModel} installed onDownload={onDownload} />);

    const button = screen.getByRole("button", { name: "Installed" });
    expect(button.disabled).toBe(true);
    fireEvent.click(button);
    expect(onDownload).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Download" })).toBeNull();
  });

  it("still offers Download for a model that is not installed", () => {
    const onDownload = vi.fn();
    render(<ExploreModelCard model={baseModel} onDownload={onDownload} />);

    const button = screen.getByRole("button", { name: "Download" });
    expect(button.disabled).toBe(false);
    fireEvent.click(button);
    expect(onDownload).toHaveBeenCalledWith(baseModel);
  });
});

describe("ExploreModelCard very small model note (#381)", () => {
  const noteId = "small-model-note";

  it.each([0.6, 1.7, 0.27])("shows the note for a %sB catalog model", (paramSize) => {
    render(<ExploreModelCard model={{ ...baseModel, param_size: paramSize }} />);
    const note = screen.getByTestId(noteId);
    expect(note.textContent).toMatch(/Very small model/);
    expect(note.textContent).toMatch(/tool use, knowledge-base search and multi-step reasoning/);
    expect(note.textContent).toMatch(/below ~4B/);
  });

  it("falls back to the parameters string when param_size is unmeasured", () => {
    render(
      <ExploreModelCard model={{ ...baseModel, param_size: undefined, parameters: "0.6B" }} />
    );
    expect(screen.getByTestId(noteId)).toBeTruthy();
  });

  it.each([4, 7])("does not show the note for a %sB model", (paramSize) => {
    render(<ExploreModelCard model={{ ...baseModel, param_size: paramSize }} />);
    expect(screen.queryByTestId(noteId)).toBeNull();
  });

  it("does not show the note when the size is unknown", () => {
    render(<ExploreModelCard model={{ ...baseModel, param_size: undefined }} />);
    expect(screen.queryByTestId(noteId)).toBeNull();
  });
});

describe("ExploreModelCard publication date (#561)", () => {
  it("shows the month and year when the date is known", () => {
    render(<ExploreModelCard model={{ ...baseModel, publishedAt: "2026-03-04T10:00:00+00:00" }} />);
    expect(screen.getByText("Published March 2026")).toBeTruthy();
  });

  it("shows nothing when the date is unknown", () => {
    render(<ExploreModelCard model={{ ...baseModel, publishedAt: null }} />);
    expect(screen.queryByText(/Published/)).toBeNull();
  });
});
