import { describe, it, expect } from "vitest";
import {
  baseName,
  decodeMarkerPath,
  getAttachmentNames,
  getDisplayContent,
  getImagePaths,
} from "./messageContent";

// #136 — stored message content can carry internal attachment markers
// ([image], [image_path:<path>]) that must never leak into what the user
// reads or copies. getDisplayContent is the single source of truth for the
// readable text, shared by the chat display and the copy button.

describe("getDisplayContent", () => {
  it("strips [image_path:…] and [image] markers and keeps the readable text", () => {
    expect(getDisplayContent("[image_path:/a/b.png][image]Describe the picture")).toBe(
      "Describe the picture"
    );
  });

  it("trims whitespace left behind by removed markers", () => {
    expect(getDisplayContent("What is this? [image_path:/tmp/photo 1.jpg]")).toBe("What is this?");
  });

  it("returns content without markers unchanged", () => {
    expect(getDisplayContent("Plain question about Paris")).toBe("Plain question about Paris");
  });

  it("keeps the error-message formatting used by the chat display", () => {
    expect(getDisplayContent("[ERROR_MESSAGE_SYSTEM] Something broke")).toBe("❌ Something broke");
  });
});

// #492 — attached documents persist as [file_path:<path>] markers. The names
// come back as chips on reload; the paths never reach the readable text.

describe("document attachment markers (#492)", () => {
  it("strips [file_path:…] from the readable text", () => {
    expect(getDisplayContent("Summarize this [file_path:/Users/me/docs/report.pdf]")).toBe(
      "Summarize this"
    );
  });

  it("recovers the attached names, in order", () => {
    expect(
      getAttachmentNames("Compare them [file_path:/a/one.pdf] [file_path:C:\\docs\\two.xlsx]")
    ).toEqual(["one.pdf", "two.xlsx"]);
  });

  it("returns no names when the message carries no attachment", () => {
    expect(getAttachmentNames("Plain question")).toEqual([]);
    expect(getAttachmentNames(undefined)).toEqual([]);
  });

  it("takes the last segment of a POSIX or Windows path", () => {
    expect(baseName("/Users/me/docs/report.pdf")).toBe("report.pdf");
    expect(baseName("C:\\Users\\me\\dossier")).toBe("dossier");
    expect(baseName("")).toBe("");
  });
});

// A real path can hold "]" and "%", which a bracket-delimited marker cannot
// carry verbatim. The backend percent-encodes both when it writes the marker
// ("%" first, so decoding is unambiguous); every reader here decodes.

describe("marker paths carrying ] and %", () => {
  // STORED is byte-for-byte what the backend writes for REAL; the pair is
  // pinned on the other side by test_marker_paths_encode_the_closing_bracket
  // in backend/tests/test_conversations.py, so the two halves cannot drift.
  const STORED = "/docs/[2026%5D 100%25/report.pdf";
  const REAL = "/docs/[2026] 100%/report.pdf";

  it("decodes an encoded marker path back to the real one", () => {
    expect(decodeMarkerPath(STORED)).toBe(REAL);
  });

  it("keeps a literal %5D in a path distinct from an encoded ]", () => {
    // A path whose name really contains "%5D" is stored as "%255D".
    expect(decodeMarkerPath("/docs/%255D.txt")).toBe("/docs/%5D.txt");
  });

  it("strips the whole marker, leaving nothing of the path behind", () => {
    expect(getDisplayContent(`Summarize this [file_path:${STORED}]`)).toBe("Summarize this");
    expect(getDisplayContent(`Look [image_path:${STORED}]`)).toBe("Look");
  });

  it("recovers the real name from an encoded attachment marker", () => {
    expect(getAttachmentNames(`Read it [file_path:${STORED}]`)).toEqual(["report.pdf"]);
    // The bracket can live in the file name itself.
    expect(getAttachmentNames("Read it [file_path:/docs/[2026%5D 100%25.pdf]")).toEqual([
      "[2026] 100%.pdf",
    ]);
  });

  it("recovers the real image path for rehydration", () => {
    expect(getImagePaths(`Look [image_path:${STORED}]`)).toEqual([REAL]);
    expect(getImagePaths("Look [image]")).toEqual([]);
  });
});
