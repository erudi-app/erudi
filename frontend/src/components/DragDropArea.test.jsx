// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import DragDropArea from "./DragDropArea.jsx";

/* --------------------------- entry-API test doubles --------------------------- */

const makeFile = (name) => new File(["content"], name, { type: "" });

// FileSystemFileEntry stub — invokes the success callback synchronously.
const makeFileEntry = (file) => ({
  isFile: true,
  isDirectory: false,
  name: file.name,
  file: (success) => success(file),
});

// FileSystemDirectoryEntry stub whose reader yields children in fixed-size
// BATCHES, then an empty array — exercises the readEntries-until-empty loop.
const makeDirEntry = (children, batchSize = 100) => ({
  isFile: false,
  isDirectory: true,
  createReader: () => {
    let idx = 0;
    return {
      readEntries: (success) => {
        const batch = children.slice(idx, idx + batchSize);
        idx += batch.length;
        success(batch);
      },
    };
  },
});

// DataTransferItem stub carrying an entry (kind "file").
const makeItem = (entry) => ({ kind: "file", webkitGetAsEntry: () => entry });

const getDropZone = (container) => container.querySelector("[data-drag-drop-area]");
const getFileInput = (container) => container.querySelector("input[type='file'][accept]");

const fireDrop = (node, { items, files = [] }) =>
  fireEvent.drop(node, { dataTransfer: { items, files } });

/* --------------------------------- lifecycle ---------------------------------- */

beforeEach(() => {
  // Default: resolve every File to a fake absolute path derived from its name.
  window.electron = { getFilePath: vi.fn((file) => `/abs/${file.name}`) };
});

afterEach(() => {
  cleanup();
  delete window.electron;
  vi.restoreAllMocks();
});

/* ----------------------------------- tests ------------------------------------ */

describe("DragDropArea — path-less entries (gap 2)", () => {
  it("drops entries whose resolved path is empty and warns, never forwarding them", async () => {
    // b.pdf resolves to an empty path — it must be filtered out.
    window.electron.getFilePath = vi.fn((file) =>
      file.name === "b.pdf" ? "" : `/abs/${file.name}`
    );
    const onFilesAdded = vi.fn();
    const { container } = render(<DragDropArea onFilesAdded={onFilesAdded} />);

    fireDrop(getDropZone(container), {
      items: [
        makeItem(makeFileEntry(makeFile("a.pdf"))),
        makeItem(makeFileEntry(makeFile("b.pdf"))),
      ],
    });

    await waitFor(() => expect(onFilesAdded).toHaveBeenCalled());
    expect(onFilesAdded).toHaveBeenCalledWith(["/abs/a.pdf"]);
    expect(screen.getByText(/couldn't resolve a path for 1 item/i)).toBeTruthy();
  });

  it("also rejects whitespace-only paths", async () => {
    window.electron.getFilePath = vi.fn((file) =>
      file.name === "b.pdf" ? "   " : `/abs/${file.name}`
    );
    const onFilesAdded = vi.fn();
    const { container } = render(<DragDropArea onFilesAdded={onFilesAdded} />);

    fireDrop(getDropZone(container), {
      items: [
        makeItem(makeFileEntry(makeFile("a.pdf"))),
        makeItem(makeFileEntry(makeFile("b.pdf"))),
      ],
    });

    await waitFor(() => expect(onFilesAdded).toHaveBeenCalledWith(["/abs/a.pdf"]));
    expect(screen.getByText(/couldn't resolve a path/i)).toBeTruthy();
  });
});

describe("DragDropArea — folder drop expansion (gap 1)", () => {
  it("recursively expands a dropped folder, looping across readEntries batches", async () => {
    const onFilesAdded = vi.fn();
    const { container } = render(<DragDropArea onFilesAdded={onFilesAdded} />);

    // A folder with a nested subfolder; batchSize 2 forces multiple readEntries
    // calls at the top level (3 children -> [a,b], [c], []).
    const sub = makeDirEntry([makeFileEntry(makeFile("deep.txt"))], 2);
    const root = makeDirEntry(
      [makeFileEntry(makeFile("a.pdf")), makeFileEntry(makeFile("b.md")), sub],
      2
    );

    fireDrop(getDropZone(container), { items: [makeItem(root)] });

    await waitFor(() => expect(onFilesAdded).toHaveBeenCalled());
    const paths = onFilesAdded.mock.calls.at(-1)[0];
    expect(paths).toEqual(expect.arrayContaining(["/abs/a.pdf", "/abs/b.md", "/abs/deep.txt"]));
    expect(paths).toHaveLength(3);
  });

  it("filters unsupported files found inside a dropped folder and warns", async () => {
    const onFilesAdded = vi.fn();
    const { container } = render(<DragDropArea onFilesAdded={onFilesAdded} />);

    const folder = makeDirEntry([
      makeFileEntry(makeFile("keep.pdf")),
      makeFileEntry(makeFile("photo.png")),
      makeFileEntry(makeFile("notes.txt")),
    ]);

    fireDrop(getDropZone(container), { items: [makeItem(folder)] });

    await waitFor(() => expect(onFilesAdded).toHaveBeenCalled());
    expect(onFilesAdded).toHaveBeenCalledWith(["/abs/keep.pdf", "/abs/notes.txt"]);
    expect(screen.getByText(/1 unsupported file skipped/i)).toBeTruthy();
  });
});

describe("DragDropArea — safety cap", () => {
  it("caps a large folder at 500 files and warns about the overflow", async () => {
    const onFilesAdded = vi.fn();
    const { container } = render(<DragDropArea onFilesAdded={onFilesAdded} />);

    const children = Array.from({ length: 501 }, (_, i) => makeFileEntry(makeFile(`f${i}.txt`)));
    const folder = makeDirEntry(children, 100); // also multi-batch

    fireDrop(getDropZone(container), { items: [makeItem(folder)] });

    await waitFor(() => expect(onFilesAdded).toHaveBeenCalled());
    const paths = onFilesAdded.mock.calls.at(-1)[0];
    expect(paths).toHaveLength(500);
    expect(screen.getByText(/capped at 500 files/i)).toBeTruthy();
  }, 15000);
});

describe("DragDropArea — plain file drop/pick unchanged", () => {
  it("handles a plain multi-file drop (no entry API) via dataTransfer.files", async () => {
    const onFilesAdded = vi.fn();
    const { container } = render(<DragDropArea onFilesAdded={onFilesAdded} />);

    // No `items` -> falls back to dataTransfer.files.
    fireDrop(getDropZone(container), {
      items: undefined,
      files: [makeFile("report.pdf"), makeFile("data.csv")],
    });

    await waitFor(() => expect(onFilesAdded).toHaveBeenCalled());
    expect(onFilesAdded).toHaveBeenCalledWith(["/abs/report.pdf", "/abs/data.csv"]);
    // Clean batch -> no warning surfaced.
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("handles a plain file pick via the hidden <input>", async () => {
    const onFilesAdded = vi.fn();
    const { container } = render(<DragDropArea onFilesAdded={onFilesAdded} />);

    const input = getFileInput(container);
    Object.defineProperty(input, "files", {
      value: [makeFile("a.txt"), makeFile("b.docx")],
      configurable: true,
    });
    fireEvent.change(input);

    await waitFor(() => expect(onFilesAdded).toHaveBeenCalled());
    expect(onFilesAdded).toHaveBeenCalledWith(["/abs/a.txt", "/abs/b.docx"]);
  });
});
