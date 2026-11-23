// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { isCountableError } from "../utils/bugCounter";
import { DOWNLOAD_CANCELLED } from "../utils/downloadStatus";

// A download that fails because of the machine (no network) is no bug: the
// card's record of it must not count on Diagnostics' bug badge, in any
// interface language (#780). The record names the failure's code and carries
// the backend's English message, never the translated sentence alone.

const { opened } = vi.hoisted(() => ({ opened: { options: null } }));

vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({
    open: (model, options) => {
      opened.options = options;
    },
  }),
}));

import ModelCard from "./ModelCard.jsx";

let sent;

beforeEach(() => {
  sent = [];
  window.logAPI = { send: (entry) => sent.push(entry) };
  opened.options = null;
  vi.spyOn(console, "error").mockImplementation(() => {});
});
afterEach(() => {
  cleanup();
  delete window.logAPI;
  vi.restoreAllMocks();
});

// What Diagnostics reads of a renderer record: its message and its data.
const asDiagnosticsEntry = (entry) => ({
  level: entry.level.toUpperCase(),
  message: [entry.msg, entry.data].filter(Boolean).join(" "),
});

const startDownload = () => {
  render(<ModelCard model={{ id: 3, name: "Qwen2.5 7B" }} type="add" />);
  fireEvent.click(screen.getByText("Add New Model"));
  expect(opened.options).toBeTruthy();
};

describe("ModelCard download failure record (#780)", () => {
  it("does not count an offline failure as a bug", () => {
    startDownload();
    opened.options.onError({
      message: "Le téléchargement s'est arrêté faute de connexion internet.",
      code: "offline",
      error_message: "you appear to be offline - check your connection and retry",
    });

    const records = sent.filter((e) => e.ns === "ModelCard");
    expect(records).toHaveLength(1);
    expect(records[0].msg).toContain("(code=offline)");
    expect(records[0].data).toContain("you appear to be offline");
    expect(isCountableError(asDiagnosticsEntry(records[0]))).toBe(false);
  });

  it("counts a failure that is Erudi's own", () => {
    startDownload();
    opened.options.onError({
      message: "Le téléchargement a échoué de manière inattendue",
      code: "unexpected",
      error_message: "KeyError: 'siblings'",
    });

    const [record] = sent.filter((e) => e.ns === "ModelCard");
    expect(isCountableError(asDiagnosticsEntry(record))).toBe(true);
  });

  it("writes nothing for a cancel, which is no failure", () => {
    startDownload();
    opened.options.onError(DOWNLOAD_CANCELLED);
    expect(sent.filter((e) => e.ns === "ModelCard" && e.level === "error")).toHaveLength(0);
  });
});
