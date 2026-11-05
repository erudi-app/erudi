// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// Slash commands in the composer (#585): a drop-up lists the commands the page
// offers, the keyboard drives it, an exact command never reaches the model,
// and running one keeps the attachments for the next message.

import QuestionInput from "./QuestionInput.jsx";
import { COMMAND_SCOPES, commandsFor, commandsOutside } from "../utils/slashCommands";

const COMMANDS = commandsFor(COMMAND_SCOPES.CONVERSATION);

const pdfFile = () =>
  new File([new Uint8Array([37, 80, 68, 70])], "report.pdf", { type: "application/pdf" });

const textarea = () => screen.getByRole("combobox");
const type = (value) => fireEvent.change(textarea(), { target: { value } });
const key = (k, extra = {}) => fireEvent.keyDown(textarea(), { key: k, ...extra });
const options = () => screen.queryAllByRole("option");
const optionNames = () => options().map((o) => o.getAttribute("data-command"));

let onSend;
let onCommand;

const renderComposer = (props = {}) =>
  render(
    <QuestionInput
      placeholder="ask"
      onSend={onSend}
      onCommand={onCommand}
      commands={COMMANDS}
      {...props}
    />
  );

beforeEach(() => {
  onSend = vi.fn();
  onCommand = vi.fn();
  window.electron = { getFilePath: (file) => `/docs/${file.name}` };
});

afterEach(() => {
  cleanup();
  delete window.electron;
});

describe("combobox semantics", () => {
  it("exposes the composer as a combobox whose expanded state follows the drop-up", () => {
    renderComposer();
    const box = textarea();
    expect(box.tagName).toBe("TEXTAREA");
    expect(box.getAttribute("aria-expanded")).toBe("false");
    expect(box.getAttribute("aria-controls")).toBeNull();
    type("/");
    expect(box.getAttribute("aria-expanded")).toBe("true");
    expect(box.getAttribute("aria-controls")).toBe(screen.getByRole("listbox").id);
    expect(box.getAttribute("aria-activedescendant")).toBe(options()[0].id);
    key("Escape");
    expect(box.getAttribute("aria-expanded")).toBe("false");
  });

  it("stays a plain textbox on a page without commands (the Arena)", () => {
    render(<QuestionInput placeholder="ask" onSend={onSend} />);
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.getByRole("textbox").getAttribute("aria-expanded")).toBeNull();
  });
});

describe("the drop-up", () => {
  it("opens on a leading slash with every command and its description", () => {
    renderComposer();
    type("/");
    const listbox = screen.getByRole("listbox");
    expect(listbox).toBeTruthy();
    expect(optionNames()).toEqual(["config", "context", "compact", "effort", "model"]);
    expect(screen.getByText("/config")).toBeTruthy();
    expect(screen.getByText("Adjust this conversation's settings")).toBeTruthy();
  });

  it("narrows as I type", () => {
    renderComposer();
    type("/m");
    expect(optionNames()).toEqual(["model"]);
  });

  it("points aria-activedescendant at the highlighted option, the first by default", () => {
    renderComposer();
    type("/");
    const first = options()[0];
    expect(first.getAttribute("aria-selected")).toBe("true");
    expect(textarea().getAttribute("aria-activedescendant")).toBe(first.id);
    expect(textarea().getAttribute("aria-controls")).toBe(screen.getByRole("listbox").id);
  });

  it("moves the highlight with Arrow Down / Up, wrapping around", () => {
    renderComposer();
    type("/");
    key("ArrowDown");
    expect(options()[1].getAttribute("aria-selected")).toBe("true");
    key("ArrowUp");
    key("ArrowUp");
    expect(options()[4].getAttribute("aria-selected")).toBe("true");
    expect(textarea().getAttribute("aria-activedescendant")).toBe(options()[4].id);
  });

  it("runs the highlighted command on Enter for a partial name, and clears the composer", () => {
    renderComposer();
    type("/");
    key("ArrowDown");
    key("Enter");
    expect(onCommand).toHaveBeenCalledWith({ name: "context", arg: null });
    expect(onSend).not.toHaveBeenCalled();
    expect(textarea().value).toBe("");
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("runs the typed command on Enter when it is already complete, never the highlight", () => {
    renderComposer();
    type("/model");
    key("Enter");
    expect(onCommand).toHaveBeenCalledWith({ name: "model", arg: null });
  });

  it("completes the highlighted name on Tab without running it", () => {
    renderComposer();
    type("/e");
    key("Tab");
    expect(textarea().value).toBe("/effort");
    expect(onCommand).not.toHaveBeenCalled();
  });

  it("closes on Escape and leaves the text alone", () => {
    renderComposer();
    type("/con");
    key("Escape");
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(textarea().value).toBe("/con");
    expect(textarea().getAttribute("aria-activedescendant")).toBeNull();
  });

  it("closes once a space follows a complete name", () => {
    renderComposer();
    type("/config");
    expect(screen.getByRole("listbox")).toBeTruthy();
    type("/config ");
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("runs a clicked command", () => {
    renderComposer();
    type("/");
    fireEvent.click(options()[4]);
    expect(onCommand).toHaveBeenCalledWith({ name: "model", arg: null });
  });

  it("does nothing on Enter while an input method is composing", () => {
    renderComposer();
    type("/");
    key("Enter", { isComposing: true });
    expect(onCommand).not.toHaveBeenCalled();
    expect(onSend).not.toHaveBeenCalled();
    type("ni hao");
    key("Enter", { isComposing: true });
    expect(onSend).not.toHaveBeenCalled();
    expect(textarea().value).toBe("ni hao");
  });

  it("never opens on a page without commands (the Arena)", () => {
    render(<QuestionInput placeholder="ask" onSend={onSend} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "/" } });
    expect(screen.queryByRole("listbox")).toBeNull();
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });
    expect(onSend).toHaveBeenCalledWith("/", [], [], []);
  });
});

describe("sending an exact command", () => {
  it("runs /config global as typed, with its argument", () => {
    renderComposer();
    type("/config global");
    key("Enter");
    expect(onCommand).toHaveBeenCalledWith({ name: "config", arg: "global" });
    expect(onSend).not.toHaveBeenCalled();
    expect(textarea().value).toBe("");
  });

  it("runs a command from the send button too", () => {
    renderComposer();
    type("/effort");
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onCommand).toHaveBeenCalledWith({ name: "effort", arg: null });
    expect(onSend).not.toHaveBeenCalled();
  });

  it.each(["please /config this", "/foo", "/config everything", "/effort high"])(
    "sends %j to the model as a normal message",
    (text) => {
      renderComposer();
      type(text);
      fireEvent.click(screen.getByLabelText("Send"));
      expect(onCommand).not.toHaveBeenCalled();
      expect(onSend).toHaveBeenCalledWith(text, [], [], []);
    }
  );

  it.each(["/compact", "/context", "  /context  "])(
    "sends nothing for %j on a page that does not offer it, and says so",
    (text) => {
      const onUnavailableCommand = vi.fn();
      renderComposer({
        commands: commandsFor(COMMAND_SCOPES.DRAFT),
        unavailableCommands: commandsOutside(COMMAND_SCOPES.DRAFT),
        onUnavailableCommand,
      });
      type(text);
      expect(screen.queryByRole("listbox")).toBeNull();
      key("Enter");
      expect(onSend).not.toHaveBeenCalled();
      expect(onCommand).not.toHaveBeenCalled();
      expect(onUnavailableCommand).toHaveBeenCalledWith({ name: text.trim().slice(1), arg: null });
      expect(textarea().value).toBe("");
    }
  );

  it("keeps the attachments when a command the page does not offer sends nothing", () => {
    renderComposer({
      commands: commandsFor(COMMAND_SCOPES.DRAFT),
      unavailableCommands: commandsOutside(COMMAND_SCOPES.DRAFT),
      onUnavailableCommand: vi.fn(),
    });
    fireEvent.change(document.querySelector('input[type="file"]'), {
      target: { files: [pdfFile()] },
    });
    type("/compact");
    key("Enter");
    expect(screen.getByText("report.pdf")).toBeTruthy();

    type("Summarize it");
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSend).toHaveBeenCalledWith("Summarize it", [], [], ["/docs/report.pdf"]);
  });

  it("never lists a command the page does not offer", () => {
    renderComposer({
      commands: commandsFor(COMMAND_SCOPES.DRAFT),
      unavailableCommands: commandsOutside(COMMAND_SCOPES.DRAFT),
      onUnavailableCommand: vi.fn(),
    });
    type("/");
    expect(optionNames()).toEqual(["config", "effort", "model"]);
    type("/co");
    expect(optionNames()).toEqual(["config"]);
  });

  it("sends a command the page does not offer with an argument as a normal message", () => {
    const onUnavailableCommand = vi.fn();
    renderComposer({
      commands: commandsFor(COMMAND_SCOPES.DRAFT),
      unavailableCommands: commandsOutside(COMMAND_SCOPES.DRAFT),
      onUnavailableCommand,
    });
    type("/compact now");
    key("Enter");
    expect(onUnavailableCommand).not.toHaveBeenCalled();
    expect(onSend).toHaveBeenCalledWith("/compact now", [], [], []);
  });

  it("keeps the attachments for the next message", () => {
    renderComposer();
    fireEvent.change(document.querySelector('input[type="file"]'), {
      target: { files: [pdfFile()] },
    });
    type("/config");
    key("Enter");
    expect(onCommand).toHaveBeenCalledTimes(1);
    expect(screen.getByText("report.pdf")).toBeTruthy();

    type("Summarize it");
    fireEvent.click(screen.getByLabelText("Send"));
    expect(onSend).toHaveBeenCalledWith("Summarize it", [], [], ["/docs/report.pdf"]);
  });
});
