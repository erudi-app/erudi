// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// #136 — a pasted (clipboard) image has no source path, so it used to be stored
// as a bare [image] placeholder and vanished on reload. The composer now persists
// the bytes via window.imageAPI.savePasted and attaches the returned real path, so
// it flows through the same [image_path:...] persistence + reload pipeline as a
// file attachment.

import QuestionInput from "./QuestionInput.jsx";

const FAKE_PATH = "C:\\Users\\me\\AppData\\Local\\erudi\\pasted-images\\paste-1.png";

// Dispatch a native "paste" carrying a clipboard image file. React reads
// e.clipboardData off the native event, so defining it here is enough.
const pasteImage = (node, file) => {
  const event = new Event("paste", { bubbles: true, cancelable: true });
  Object.defineProperty(event, "clipboardData", {
    value: { items: [{ kind: "file", type: file.type, getAsFile: () => file }] },
  });
  fireEvent(node, event);
};

const pngFile = () =>
  new File([new Uint8Array([137, 80, 78, 71])], "pasted.png", { type: "image/png" });

const svgFile = () => new File(["<svg/>"], "vector.svg", { type: "image/svg+xml" });

beforeEach(() => {
  window.imageAPI = { savePasted: vi.fn(async () => FAKE_PATH) };
});

afterEach(() => {
  cleanup();
  delete window.imageAPI;
  delete window.electron;
});

describe("QuestionInput pasted-image persistence (#136)", () => {
  it("persists a pasted image and attaches it as a path-attached image", async () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} canAttachImages />);

    pasteImage(screen.getByPlaceholderText("ask"), pngFile());

    // The save IPC is called with the image's data URL.
    await waitFor(() => expect(window.imageAPI.savePasted).toHaveBeenCalledTimes(1));
    const dataUrl = window.imageAPI.savePasted.mock.calls[0][0];
    expect(dataUrl).toMatch(/^data:image\/png;base64,/);

    // The composer shows the attached thumbnail (path-attached image held).
    await screen.findByAltText("attachment 1");

    fireEvent.click(screen.getByLabelText("Send"));

    // Sent with the persisted real path in the parallel imagePaths array.
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith("", [dataUrl], [FAKE_PATH], []);
  });

  it("does not persist a file-origin image (it already has a real path)", async () => {
    const onSend = vi.fn();
    window.electron = { getFilePath: () => "C:\\photos\\cat.png" };
    render(<QuestionInput placeholder="ask" onSend={onSend} canAttachImages />);

    const fileInput = document.querySelector('input[type="file"]');
    fireEvent.change(fileInput, { target: { files: [pngFile()] } });

    await screen.findByAltText("attachment 1");
    expect(window.imageAPI.savePasted).not.toHaveBeenCalled();

    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSend).toHaveBeenCalledWith(
      "",
      [expect.stringMatching(/^data:image\/png;base64,/)],
      ["C:\\photos\\cat.png"],
      []
    );
  });

  it("collects nothing from a paste when the model is not vision-capable", async () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} canAttachImages={false} />);

    pasteImage(screen.getByPlaceholderText("ask"), pngFile());

    // Give any async reader/persist a chance to (not) run.
    await Promise.resolve();
    expect(window.imageAPI.savePasted).not.toHaveBeenCalled();
    expect(screen.queryByAltText("attachment 1")).toBeNull();
  });
});

describe("QuestionInput image-pipeline robustness", () => {
  it("rejects an unsupported format (SVG) with an error instead of attaching it", async () => {
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages />);
    const fileInput = document.querySelector('input[type="file"]');
    fireEvent.change(fileInput, { target: { files: [svgFile()] } });

    await screen.findByRole("alert");
    expect(screen.queryByAltText("attachment 1")).toBeNull();
  });

  it("rejects an image over the size cap", async () => {
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages />);
    const big = pngFile();
    Object.defineProperty(big, "size", { value: 21 * 1024 * 1024 });
    const fileInput = document.querySelector('input[type="file"]');
    fireEvent.change(fileInput, { target: { files: [big] } });

    await screen.findByRole("alert");
    expect(screen.queryByAltText("attachment 1")).toBeNull();
  });

  it("shows an add-another tile once an image is attached", async () => {
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages />);
    const fileInput = document.querySelector('input[type="file"]');
    fireEvent.change(fileInput, { target: { files: [pngFile()] } });

    await screen.findByAltText("attachment 1");
    await screen.findByLabelText("Add image");
  });

  it("caps attachments at maxImages and warns when more are picked", async () => {
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages maxImages={1} />);
    const fileInput = document.querySelector('input[type="file"]');
    fireEvent.change(fileInput, { target: { files: [pngFile(), pngFile()] } });

    await screen.findByAltText("attachment 1");
    expect(screen.queryByAltText("attachment 2")).toBeNull();
    await screen.findByRole("alert");
    // At the cap, the add-another tile is hidden.
    expect(screen.queryByLabelText("Add image")).toBeNull();
  });
});
