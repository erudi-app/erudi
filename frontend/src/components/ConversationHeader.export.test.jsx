// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// Export from the conversation header (#430): the settings panel carries an
// "Export as Markdown" button, named in words like the command buttons, whenever
// the page gives it an `onExport`.

import ConversationHeader from "./ConversationHeader.jsx";
import { COMMAND_SCOPES, commandsFor } from "../utils/slashCommands";

afterEach(cleanup);

const COMMANDS = commandsFor(COMMAND_SCOPES.CONVERSATION);

const renderHeader = (props = {}) =>
  render(
    <ConversationHeader
      modelName="Qwen3-4B"
      models={[{ id: 1, name: "Qwen3-4B" }]}
      currentModelId={1}
      onPickModel={() => {}}
      commands={COMMANDS}
      onCommand={vi.fn()}
      {...props}
    />
  );

const openPanel = () =>
  fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));

describe("ConversationHeader export", () => {
  it("offers the export in the settings panel, labelled in words, with no slash name", () => {
    renderHeader({ onExport: vi.fn() });
    expect(screen.queryByRole("button", { name: "Export as Markdown" })).toBeNull();
    openPanel();
    const button = screen.getByRole("button", { name: "Export as Markdown" });
    expect(button.textContent).not.toContain("/");
  });

  it("runs the export when pressed", () => {
    const onExport = vi.fn();
    renderHeader({ onExport });
    openPanel();
    fireEvent.click(screen.getByRole("button", { name: "Export as Markdown" }));
    expect(onExport).toHaveBeenCalledTimes(1);
  });

  it("shows no export button on a page that does not offer one", () => {
    renderHeader();
    openPanel();
    expect(screen.queryByRole("button", { name: "Export as Markdown" })).toBeNull();
  });

  it("is not sent to the command runner", () => {
    const onCommand = vi.fn();
    renderHeader({ onExport: vi.fn(), onCommand });
    openPanel();
    fireEvent.click(screen.getByRole("button", { name: "Export as Markdown" }));
    expect(onCommand).not.toHaveBeenCalled();
  });
});
