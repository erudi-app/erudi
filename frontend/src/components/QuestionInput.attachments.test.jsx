// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// #492 - the composer accepts documents and folders next to images. Documents
// never travel as bytes: the renderer knows the real filesystem path and the
// backend reads the file on the same machine, so the composer only has to
// collect paths, show a chip per file, and hand them to onSend.

import QuestionInput from "./QuestionInput.jsx";

const pdfFile = (name = "report.pdf") =>
  new File([new Uint8Array([37, 80, 68, 70])], name, { type: "application/pdf" });

const zipFile = () =>
  new File([new Uint8Array([80, 75])], "archive.zip", {
    type: "application/zip",
  });

const fileInput = () => document.querySelector('input[type="file"]');

beforeEach(() => {
  window.electron = { getFilePath: (file) => `/docs/${file.name}` };
});

afterEach(() => {
  cleanup();
  delete window.electron;
  delete window.imageAPI;
});

describe("QuestionInput document attachments (#492)", () => {
  it("accepts a .pdf as a chip and sends its path", () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} />);

    fireEvent.change(fileInput(), { target: { files: [pdfFile()] } });

    expect(screen.getByText("report.pdf")).toBeTruthy();

    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSend).toHaveBeenCalledWith("", [], [], ["/docs/report.pdf"]);
  });

  it("rejects an unsupported file type instead of attaching it", () => {
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} />);

    fireEvent.change(fileInput(), { target: { files: [zipFile()] } });

    expect(screen.getByRole("alert")).toBeTruthy();
    expect(screen.queryByText("archive.zip")).toBeNull();
  });

  it("rejects a document with no path on disk", () => {
    window.electron = { getFilePath: () => "" };
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} />);

    fireEvent.change(fileInput(), { target: { files: [pdfFile()] } });

    expect(screen.getByRole("alert")).toBeTruthy();
    expect(screen.queryByText("report.pdf")).toBeNull();
  });

  it("rejects a document over the size cap", () => {
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} />);

    const big = pdfFile();
    Object.defineProperty(big, "size", { value: 51 * 1024 * 1024 });
    fireEvent.change(fileInput(), { target: { files: [big] } });

    expect(screen.getByRole("alert")).toBeTruthy();
    expect(screen.queryByText("report.pdf")).toBeNull();
  });

  it("removes an attached document from its chip", () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} />);

    fireEvent.change(fileInput(), { target: { files: [pdfFile()] } });
    fireEvent.click(screen.getByLabelText("Remove file"));

    expect(screen.queryByText("report.pdf")).toBeNull();
    // Nothing left to send: the send button stays disabled.
    expect(screen.getByLabelText("Send").disabled).toBe(true);
  });

  it("caps the number of attached documents", () => {
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} maxAttachments={1} />);

    fireEvent.change(fileInput(), {
      target: { files: [pdfFile("a.pdf"), pdfFile("b.pdf")] },
    });

    expect(screen.getByText("a.pdf")).toBeTruthy();
    expect(screen.queryByText("b.pdf")).toBeNull();
    expect(screen.getByRole("alert")).toBeTruthy();
  });

  it("accepts documents even when the model cannot read images", () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} canAttachImages={false} />);

    fireEvent.change(fileInput(), { target: { files: [pdfFile()] } });

    expect(screen.getByText("report.pdf")).toBeTruthy();
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSend).toHaveBeenCalledWith("", [], [], ["/docs/report.pdf"]);
  });

  it("attaches a dropped folder by its path and lets the backend walk it", () => {
    const onSend = vi.fn();
    render(<QuestionInput placeholder="ask" onSend={onSend} />);

    const folder = new File([], "dossier", { type: "" });
    fireEvent.drop(screen.getByPlaceholderText("ask").closest("div"), {
      dataTransfer: {
        files: [folder],
        items: [
          {
            kind: "file",
            getAsFile: () => folder,
            webkitGetAsEntry: () => ({ isDirectory: true }),
          },
        ],
      },
    });

    expect(screen.getByText("dossier")).toBeTruthy();
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSend).toHaveBeenCalledWith("", [], [], ["/docs/dossier"]);
  });

  it("clears the attachments once the question is sent", () => {
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} />);

    fireEvent.change(fileInput(), { target: { files: [pdfFile()] } });
    fireEvent.click(screen.getByLabelText("Send"));

    expect(screen.queryByText("report.pdf")).toBeNull();
  });
});
