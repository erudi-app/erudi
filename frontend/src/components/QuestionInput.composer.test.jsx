// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// Composer behaviors beyond the pasted-image persistence already pinned in
// QuestionInput.test.jsx: keyboard send semantics, image removal, the attach
// buttons opening the picker, drag-and-drop onto the composer, unreadable
// files, and the no-bridge fallbacks for pasted images.

import QuestionInput from "./QuestionInput.jsx";

const pngFile = () =>
  new File([new Uint8Array([137, 80, 78, 71])], "shot.png", { type: "image/png" });

const pasteImage = (node, file) => {
  const event = new Event("paste", { bubbles: true, cancelable: true });
  Object.defineProperty(event, "clipboardData", {
    value: { items: [{ kind: "file", type: file.type, getAsFile: () => file }] },
  });
  fireEvent(node, event);
};

const fileInput = () => document.querySelector('input[type="file"]');

afterEach(() => {
  cleanup();
  delete window.imageAPI;
  delete window.electron;
  vi.restoreAllMocks();
});

describe("QuestionInput keyboard send", () => {
  it("sends the trimmed text on Enter and clears the composer", () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} />);

    const textarea = screen.getByPlaceholderText("ask");
    fireEvent.change(textarea, { target: { value: "  hello there  " } });
    fireEvent.keyDown(textarea, { key: "Enter" });

    expect(onSend).toHaveBeenCalledWith("hello there", [], [], []);
    expect(textarea.value).toBe("");
  });

  it("inserts a newline on Shift+Enter instead of sending", () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} />);

    const textarea = screen.getByPlaceholderText("ask");
    fireEvent.change(textarea, { target: { value: "line one" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });

    expect(onSend).not.toHaveBeenCalled();
    expect(textarea.value).toBe("line one");
  });

  it("ignores Enter when there is nothing to send", () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} />);

    const textarea = screen.getByPlaceholderText("ask");
    fireEvent.change(textarea, { target: { value: "   " } });
    fireEvent.keyDown(textarea, { key: "Enter" });

    expect(onSend).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Send").disabled).toBe(true);
  });

  it("ignores a paste that carries no clipboard items", () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} />);

    const event = new Event("paste", { bubbles: true, cancelable: true });
    Object.defineProperty(event, "clipboardData", { value: {} });
    fireEvent(screen.getByPlaceholderText("ask"), event);

    expect(screen.queryByAltText("attachment 1")).toBeNull();
  });
});

describe("QuestionInput attachment management", () => {
  it("removes an attached image and clears any attach error", async () => {
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages maxImages={1} />);

    // Two picks: one attaches, the surplus raises the cap warning.
    fireEvent.change(fileInput(), { target: { files: [pngFile(), pngFile()] } });
    await screen.findByAltText("attachment 1");
    await screen.findByRole("alert");

    fireEvent.click(screen.getByLabelText("Remove image"));

    expect(screen.queryByAltText("attachment 1")).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("opens the file picker from the attach icon and the add-another tile", async () => {
    const clickSpy = vi.spyOn(HTMLInputElement.prototype, "click").mockImplementation(() => {});
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages maxImages={4} />);

    fireEvent.click(screen.getByLabelText("Attach image"));
    expect(clickSpy).toHaveBeenCalledTimes(1);

    fireEvent.change(fileInput(), { target: { files: [pngFile()] } });
    await screen.findByAltText("attachment 1");

    fireEvent.click(screen.getByLabelText("Add image"));
    expect(clickSpy).toHaveBeenCalledTimes(2);
  });

  it("accepts a drop on the composer and highlights while dragging over", async () => {
    const { container } = render(
      <QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages />
    );

    const panel = container.querySelector(".rounded-\\[20px\\]");
    fireEvent.dragOver(panel);
    expect(panel.className).toContain("border-emerald-400/60");
    fireEvent.dragLeave(panel);
    expect(panel.className).toContain("border-emerald-200/20");

    fireEvent.drop(panel, { dataTransfer: { files: [pngFile()] } });
    await screen.findByAltText("attachment 1");
  });

  it("ignores a drop that carries no files", () => {
    const { container } = render(
      <QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages />
    );

    const panel = container.querySelector(".rounded-\\[20px\\]");
    fireEvent.drop(panel, { dataTransfer: { files: [] } });
    expect(screen.queryByAltText("attachment 1")).toBeNull();
  });
});

describe("QuestionInput unreadable files", () => {
  const stubFileReader = (behavior) => {
    class StubFileReader {
      readAsDataURL() {
        queueMicrotask(() => behavior(this));
      }
    }
    vi.stubGlobal("FileReader", StubFileReader);
  };

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("surfaces an error when the reader fails", async () => {
    stubFileReader((reader) => reader.onerror());
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages />);

    fireEvent.change(fileInput(), { target: { files: [pngFile()] } });

    expect((await screen.findByRole("alert")).textContent).toMatch(/could not be read/);
    expect(screen.queryByAltText("attachment 1")).toBeNull();
  });

  it("surfaces an error when the reader yields something that is not an image data URL", async () => {
    stubFileReader((reader) => {
      reader.result = "not-a-data-url";
      reader.onload();
    });
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages />);

    fireEvent.change(fileInput(), { target: { files: [pngFile()] } });

    expect((await screen.findByRole("alert")).textContent).toMatch(/could not be read/);
    expect(screen.queryByAltText("attachment 1")).toBeNull();
  });
});

describe("QuestionInput pasted image without a persistence bridge", () => {
  it("attaches with an empty path when window.imageAPI is absent", async () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} canAttachImages />);

    pasteImage(screen.getByPlaceholderText("ask"), pngFile());
    await screen.findByAltText("attachment 1");

    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSend).toHaveBeenCalledWith("", [expect.stringMatching(/^data:image\/png/)], [""], []);
  });

  it("attaches with an empty path when the persistence call throws", async () => {
    const onSend = vi.fn();
    window.imageAPI = {
      savePasted: vi.fn(async () => {
        throw new Error("disk full");
      }),
    };
    render(<QuestionInput placeholder="ask" onSend={onSend} canAttachImages />);

    pasteImage(screen.getByPlaceholderText("ask"), pngFile());
    await screen.findByAltText("attachment 1");
    await waitFor(() => expect(window.imageAPI.savePasted).toHaveBeenCalled());

    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSend).toHaveBeenCalledWith("", [expect.stringMatching(/^data:image\/png/)], [""], []);
  });
});
