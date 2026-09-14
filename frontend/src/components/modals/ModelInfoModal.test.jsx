// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, afterEach, vi } from "vitest";
import { render, cleanup, screen, fireEvent } from "@testing-library/react";

import ModelInfoModal from "./ModelInfoModal.jsx";

afterEach(cleanup);

const richModel = {
  name: "Qwen2.5-7B-Instruct",
  description: "A capable instruct model.",
  size: "4.5 GB",
  parameters: "7B",
  author: "Qwen",
  library: "transformers",
  downloads: "1,234,567",
  likes: "890",
  lastUpdate: "2026-01-01",
  pipeline: "text-generation",
  rawMetadata: '{"license": "apache-2.0"}',
};

const setup = (modelInfo = richModel, overrides = {}) => {
  const props = {
    modelInfo,
    isOpen: true,
    onClose: vi.fn(),
    onDownload: vi.fn(),
    ...overrides,
  };
  const utils = render(<ModelInfoModal {...props} />);
  return { props, ...utils };
};

describe("ModelInfoModal", () => {
  it("renders nothing when closed or without a model", () => {
    setup(richModel, { isOpen: false });
    expect(screen.queryByText("Qwen2.5-7B-Instruct")).toBeNull();
    cleanup();

    setup(null);
    expect(screen.queryByText("Basic Info")).toBeNull();
  });

  it("shows every populated field of a rich model", () => {
    setup();
    expect(screen.getByText("Qwen2.5-7B-Instruct")).toBeTruthy();
    expect(screen.getByText("A capable instruct model.")).toBeTruthy();
    expect(screen.getByText("4.5 GB")).toBeTruthy();
    expect(screen.getByText("7B")).toBeTruthy();
    expect(screen.getByText("Qwen")).toBeTruthy();
    expect(screen.getByText("transformers")).toBeTruthy();
    expect(screen.getByText("1,234,567")).toBeTruthy();
    expect(screen.getByText("890")).toBeTruthy();
    expect(screen.getByText("2026-01-01")).toBeTruthy();
    expect(screen.getByText("text-generation")).toBeTruthy();
  });

  it("falls back on a placeholder description and hides Unknown/absent optionals", () => {
    setup({
      name: "bare-model",
      size: "1 GB",
      parameters: "1B",
      author: "Unknown",
      library: "Unknown",
      downloads: "Unknown",
      likes: undefined,
      lastUpdate: "Unknown",
      pipeline: "Unknown",
    });

    expect(screen.getByText("No description available")).toBeTruthy();
    expect(screen.queryByText("Author:")).toBeNull();
    expect(screen.queryByText("Library:")).toBeNull();
    expect(screen.queryByText("Downloads:")).toBeNull();
    expect(screen.queryByText("Likes:")).toBeNull();
    expect(screen.queryByText("Last Update:")).toBeNull();
    expect(screen.queryByText("Pipeline:")).toBeNull();
    // No rawMetadata -> no collapsible section at all.
    expect(screen.queryByText("Show Raw Metadata")).toBeNull();
  });

  it("toggles the raw metadata section", () => {
    setup();
    expect(screen.queryByText('{"license": "apache-2.0"}')).toBeNull();

    fireEvent.click(screen.getByText("Show Raw Metadata"));
    expect(screen.getByText('{"license": "apache-2.0"}')).toBeTruthy();

    fireEvent.click(screen.getByText("Show Raw Metadata"));
    // The exit animation may keep the node around briefly; the toggle state is
    // what we pin by toggling back on without error.
    fireEvent.click(screen.getByText("Show Raw Metadata"));
    expect(screen.getByText('{"license": "apache-2.0"}')).toBeTruthy();
  });

  // Context windows: the trained window is a fact of the model; the allocated
  // one exists only while this model is the loaded one, so its row only shows
  // when the backend resolved a value.
  it("shows the trained and allocated context windows when known", () => {
    setup({ ...richModel, context_window: 40960, allocated_context_window: 8192 });
    expect(screen.getByText("Context window:")).toBeTruthy();
    expect(screen.getByText("40960 tokens")).toBeTruthy();
    expect(screen.getByText("Currently allocated:")).toBeTruthy();
    expect(screen.getByText("8192 tokens")).toBeTruthy();
  });

  it("hides the allocated row when null and both rows when unknown", () => {
    setup({ ...richModel, context_window: 40960, allocated_context_window: null });
    expect(screen.getByText("Context window:")).toBeTruthy();
    expect(screen.queryByText("Currently allocated:")).toBeNull();
    cleanup();

    setup(richModel);
    expect(screen.queryByText("Context window:")).toBeNull();
    expect(screen.queryByText("Currently allocated:")).toBeNull();
  });

  it("Download hands the full model back and closes", () => {
    const { props } = setup();
    fireEvent.click(screen.getByText("Download"));

    expect(props.onDownload).toHaveBeenCalledTimes(1);
    expect(props.onDownload).toHaveBeenCalledWith(richModel);
    expect(props.onClose).toHaveBeenCalledTimes(1);
  });

  // The details modal is also opened from Installed cards and from catalog
  // cards of models already on disk; it must not offer a second download.
  it("offers Download on a model that is not installed", () => {
    setup(richModel, { installed: false });
    expect(screen.getByText("Download")).toBeTruthy();
    expect(screen.queryByText("Installed")).toBeNull();
  });

  it("shows the installed state instead of Download on an installed model", () => {
    const { props } = setup(richModel, { installed: true });
    expect(screen.queryByText("Download")).toBeNull();
    expect(screen.getByText("Installed")).toBeTruthy();

    fireEvent.click(screen.getByText("Close"));
    expect(props.onClose).toHaveBeenCalledTimes(1);
    expect(props.onDownload).not.toHaveBeenCalled();
  });

  it("Cancel and the header X close without downloading", () => {
    const { props, container } = setup();
    fireEvent.click(screen.getByText("Cancel"));
    expect(props.onClose).toHaveBeenCalledTimes(1);

    const headerX = container.querySelector(".flex-1 + button");
    fireEvent.click(headerX);
    expect(props.onClose).toHaveBeenCalledTimes(2);
    expect(props.onDownload).not.toHaveBeenCalled();
  });

  it("Escape closes the modal, and only while it is open", () => {
    const { props } = setup();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(props.onClose).toHaveBeenCalledTimes(1);
    expect(props.onDownload).not.toHaveBeenCalled();
    cleanup();

    const closed = setup(richModel, { isOpen: false });
    fireEvent.keyDown(document, { key: "Escape" });
    expect(closed.props.onClose).not.toHaveBeenCalled();
  });
});
