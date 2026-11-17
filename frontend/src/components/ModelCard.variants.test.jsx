// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// Card variants beyond the basic local actions: the "add" tile, the base-card
// download flow through the download modal, hardware gating, the orphaned KB
// assistant (weights missing -> Chat blocked, re-bind picker) and the
// metadata branches per card type.

const { openMock } = vi.hoisted(() => ({ openMock: vi.fn() }));
vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: openMock }),
}));

import ModelCard from "./ModelCard.jsx";

const baseModel = {
  name: "Llama 3.1 8B Instruct",
  size: "4.9 GB",
  downloads: "1.2M",
  likes: "3400",
  author: "meta-llama",
  library: "transformers",
};

const orphanAssistant = {
  id: 12,
  name: "Support Assistant",
  kb_id: 3,
  weights_available: false,
  link: "org/base-model",
};

afterEach(() => {
  cleanup();
});
beforeEach(() => {
  openMock.mockReset();
});

describe("ModelCard add tile", () => {
  it("invokes onDownload directly when clicked", () => {
    const onDownload = vi.fn();
    render(<ModelCard model={{ name: "unused" }} type="add" onDownload={onDownload} />);

    // The add tile has no model actions, just the invite text.
    fireEvent.click(screen.getByText("Add New Model").closest("div"));
    expect(onDownload).not.toHaveBeenCalled(); // model provided -> goes through the modal
    expect(openMock).toHaveBeenCalledTimes(1);
  });

  it("falls back to onDownload when no model is given", () => {
    const onDownload = vi.fn();
    render(<ModelCard model={undefined} type="add" onDownload={onDownload} />);

    fireEvent.click(screen.getByText("Add New Model").closest("div"));
    expect(onDownload).toHaveBeenCalledTimes(1);
    expect(openMock).not.toHaveBeenCalled();
  });
});

describe("ModelCard base card", () => {
  it("shows remote metadata and the tested-team badge", () => {
    render(<ModelCard model={baseModel} type="base" />);

    expect(screen.getByText("Size: 4.9 GB")).toBeTruthy();
    expect(screen.getByText("Downloads: 1.2M")).toBeTruthy();
    expect(screen.getByText("Likes: 3400")).toBeTruthy();
    expect(screen.getByText("Author: meta-llama")).toBeTruthy();
    expect(screen.getByText("Library: transformers")).toBeTruthy();
    expect(screen.getByTitle(/Tested by the Erudi team/)).toBeTruthy();
  });

  it("hides Unknown metadata fields", () => {
    render(
      <ModelCard
        model={{ name: "Mystery", size: "1 GB", downloads: "Unknown", author: "Unknown" }}
        type="base"
      />
    );

    expect(screen.queryByText(/Downloads:/)).toBeNull();
    expect(screen.queryByText(/Author:/)).toBeNull();
    expect(screen.queryByTitle(/Tested by the Erudi team/)).toBeNull();
  });

  it("fires onDownload for a runnable model and blocks an unrunnable one", () => {
    const onDownload = vi.fn();
    const { unmount } = render(
      <ModelCard model={{ ...baseModel, runnable: true }} type="base" onDownload={onDownload} />
    );
    fireEvent.click(screen.getByTitle("Download"));
    expect(onDownload).toHaveBeenCalledWith({ ...baseModel, runnable: true });
    unmount();

    onDownload.mockReset();
    render(
      <ModelCard model={{ ...baseModel, runnable: false }} type="base" onDownload={onDownload} />
    );
    expect(screen.getByText("Unavailable on your hardware")).toBeTruthy();
    const blocked = screen.getByTitle("Unavailable on your hardware", { selector: "button" });
    expect(blocked.disabled).toBe(true);
    fireEvent.click(blocked);
    expect(onDownload).not.toHaveBeenCalled();
  });
});

describe("ModelCard local card", () => {
  it("shows local metadata and calls onDelete / onChat / onKnowledgeBase", () => {
    const onDelete = vi.fn();
    const onChat = vi.fn();
    const onKnowledgeBase = vi.fn();
    const model = {
      id: 1,
      name: "Local Model",
      size: "1 GB",
      parameters: "1B",
      lastUpdate: "2026-01-01",
    };
    render(
      <ModelCard
        model={model}
        type="local"
        onDelete={onDelete}
        onChat={onChat}
        onKnowledgeBase={onKnowledgeBase}
      />
    );

    expect(screen.getByText("Size: 1 GB")).toBeTruthy();
    expect(screen.getByText("Parameters: 1B")).toBeTruthy();
    expect(screen.getByText("Last update: 2026-01-01")).toBeTruthy();

    fireEvent.click(screen.getByTitle("Delete model"));
    expect(onDelete).toHaveBeenCalledWith(model);
    fireEvent.click(screen.getByTitle("Chat"));
    expect(onChat).toHaveBeenCalledWith(model);
    fireEvent.click(screen.getByTitle("Knowledge Base"));
    expect(onKnowledgeBase).toHaveBeenCalledWith(model);
  });

  it("describes a healthy assistant as using its base model's weights", () => {
    render(
      <ModelCard
        model={{ id: 2, name: "Helper", kb_id: 1, weights_available: true }}
        type="local"
        baseModelName="Qwen3 0.6B"
      />
    );

    expect(screen.getByText("Uses the weights of Qwen3 0.6B")).toBeTruthy();
    expect(screen.queryByText(/Size:/)).toBeNull();
  });
});

describe("ModelCard orphaned assistant (#225/#208)", () => {
  it("blocks Chat and flags the missing weights", () => {
    const onChat = vi.fn();
    render(<ModelCard model={orphanAssistant} type="local" onChat={onChat} />);

    expect(screen.getByText("Weights missing")).toBeTruthy();
    expect(screen.getByText("Model weights missing")).toBeTruthy();

    const chatButton = screen.getByTitle("Model weights missing - re-bind to chat");
    expect(chatButton.disabled).toBe(true);
    fireEvent.click(chatButton);
    expect(onChat).not.toHaveBeenCalled();
  });

  it("re-binds through the picker and reports the chosen target", () => {
    const onRebind = vi.fn();
    const targets = [
      { id: 1, name: "Base A" },
      { id: 2, name: "Base B" },
    ];
    render(
      <ModelCard model={orphanAssistant} type="local" rebindTargets={targets} onRebind={onRebind} />
    );

    fireEvent.click(screen.getByTitle("Re-bind to another installed model"));
    fireEvent.click(screen.getByText("Base B"));

    expect(onRebind).toHaveBeenCalledWith(orphanAssistant, targets[1]);
    // Picking a target closes the picker.
    expect(screen.queryByText("Base A")).toBeNull();
  });

  it("explains when no installed base model is available to re-bind to", () => {
    render(<ModelCard model={orphanAssistant} type="local" rebindTargets={[]} />);

    fireEvent.click(screen.getByTitle("Re-bind to another installed model"));
    expect(screen.getByText("No installed base model available")).toBeTruthy();
  });
});

describe("ModelCard publication date (#561)", () => {
  it("shows the month and year on a catalog card", () => {
    render(
      <ModelCard model={{ ...baseModel, publishedAt: "2026-03-04T10:00:00+00:00" }} type="base" />
    );
    expect(screen.getByText("Published March 2026")).toBeTruthy();
  });

  it("shows nothing when the date is unknown", () => {
    render(<ModelCard model={{ ...baseModel, publishedAt: null }} type="base" />);
    expect(screen.queryByText(/Published/)).toBeNull();
  });
});
