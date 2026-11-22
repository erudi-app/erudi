// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// #760 - a text file renamed .png used to be attached as an image and failed
// at send. The composer now judges a file by its bytes: the signature must be
// a supported format and the browser must decode it, and the data URL carries
// the MIME the bytes announce, not the declared one.

import QuestionInput from "./QuestionInput.jsx";

const PNG_HEADER = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0];
const JPEG_HEADER = [0xff, 0xd8, 0xff, 0xe0, 0, 0x10, 0x4a, 0x46, 0x49, 0x46, 0, 1];

const fileInput = () => document.querySelector('input[type="file"]');

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  delete window.electron;
});

describe("QuestionInput images are checked by content (#760)", () => {
  it("refuses a text file renamed .png and shows no thumbnail", async () => {
    const decode = vi.fn(async () => ({ close: vi.fn() }));
    vi.stubGlobal("createImageBitmap", decode);
    const fake = new File(["Note: this is not an image"], "fake.png", { type: "image/png" });
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages />);

    fireEvent.change(fileInput(), { target: { files: [fake] } });

    expect((await screen.findByRole("alert")).textContent).toMatch(/could not be read/);
    expect(screen.queryByAltText("attachment 1")).toBeNull();
    expect(decode).not.toHaveBeenCalled();
  });

  it("refuses a PNG-looking file the browser cannot decode", async () => {
    const decode = vi.fn(async () => {
      throw new Error("InvalidStateError");
    });
    vi.stubGlobal("createImageBitmap", decode);
    const broken = new File([new Uint8Array(PNG_HEADER)], "broken.png", { type: "image/png" });
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages />);

    fireEvent.change(fileInput(), { target: { files: [broken] } });

    expect((await screen.findByRole("alert")).textContent).toMatch(/could not be read/);
    expect(decode).toHaveBeenCalledTimes(1);
    expect(screen.queryByAltText("attachment 1")).toBeNull();
  });

  it("attaches a decodable image under the MIME its bytes announce", async () => {
    const close = vi.fn();
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn(async () => ({ close }))
    );
    window.electron = { getFilePath: () => "/pics/real.png" };
    const jpegNamedPng = new File([new Uint8Array(JPEG_HEADER)], "real.png", {
      type: "image/png",
    });
    render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages />);

    fireEvent.change(fileInput(), { target: { files: [jpegNamedPng] } });

    const thumb = await screen.findByAltText("attachment 1");
    expect(thumb.getAttribute("src")).toMatch(/^data:image\/jpeg;base64,/);
    expect(close).toHaveBeenCalledTimes(1);
  });

  describe("a mixed batch", () => {
    const fake = () => new File(["Note"], "fake.png", { type: "image/png" });
    const png = (name) => new File([new Uint8Array(PNG_HEADER)], name, { type: "image/png" });
    const jpeg = (name) => new File([new Uint8Array(JPEG_HEADER)], name, { type: "image/jpeg" });

    it("takes the slots from the verified files, in the order picked", async () => {
      vi.stubGlobal(
        "createImageBitmap",
        vi.fn(async () => ({ close: vi.fn() }))
      );
      render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages maxImages={2} />);

      fireEvent.change(fileInput(), { target: { files: [fake(), png("a.png"), jpeg("b.jpg")] } });

      const first = await screen.findByAltText("attachment 1");
      const second = await screen.findByAltText("attachment 2");
      expect(first.getAttribute("src")).toMatch(/^data:image\/png;base64,/);
      expect(second.getAttribute("src")).toMatch(/^data:image\/jpeg;base64,/);
      expect((await screen.findByRole("alert")).textContent).toMatch(/could not be read/);
    });

    it("reports the budget, not the unreadable file, when verified files were dropped", async () => {
      vi.stubGlobal(
        "createImageBitmap",
        vi.fn(async () => ({ close: vi.fn() }))
      );
      render(<QuestionInput placeholder="ask" onSend={vi.fn()} canAttachImages maxImages={2} />);

      fireEvent.change(fileInput(), {
        target: { files: [fake(), png("a.png"), png("b.png"), png("c.png")] },
      });

      await screen.findByAltText("attachment 2");
      expect((await screen.findByRole("alert")).textContent).not.toMatch(/could not be read/);
      expect(screen.queryByAltText("attachment 3")).toBeNull();
    });
  });
});
