// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// The conversation header (#585): a static line naming the model and a
// settings icon opening one button per command the page offers. Later
// commands and the unload button slot in without changing its structure.

import ConversationHeader from "./ConversationHeader.jsx";
import { COMMAND_SCOPES, commandsFor } from "../utils/slashCommands";

afterEach(cleanup);

const COMMANDS = commandsFor(COMMAND_SCOPES.CONVERSATION);

const openPanel = () =>
  fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));

describe("ConversationHeader", () => {
  it("reads a static line naming the model, with no dropdown", () => {
    render(<ConversationHeader modelName="Qwen3-4B" commands={COMMANDS} onCommand={vi.fn()} />);
    expect(screen.getByRole("heading").textContent).toBe("You are chatting with Qwen3-4B");
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("opens a panel with one button per command, each running it", () => {
    const onCommand = vi.fn();
    render(<ConversationHeader modelName="Qwen3-4B" commands={COMMANDS} onCommand={onCommand} />);
    expect(screen.queryByRole("button", { name: /Config/ })).toBeNull();
    openPanel();
    const buttons = ["Config", "Effort", "Model"].map((label) =>
      screen.getByRole("button", { name: new RegExp(`^${label}`) })
    );
    fireEvent.click(buttons[0]);
    fireEvent.click(buttons[2]);
    expect(onCommand.mock.calls).toEqual([
      [{ name: "config", arg: null }],
      [{ name: "model", arg: null }],
    ]);
  });

  it("orders the panel's buttons Model, Config, Effort", () => {
    render(<ConversationHeader modelName="Qwen3-4B" commands={COMMANDS} onCommand={vi.fn()} />);
    openPanel();
    const labels = screen
      .getAllByRole("button")
      .map((b) => b.textContent)
      .filter((text) => text.includes("/"));
    expect(labels).toEqual(["Model/model", "Config/config", "Effort/effort"]);
  });

  it("disables a command the page says is unavailable", () => {
    const onCommand = vi.fn();
    render(
      <ConversationHeader
        modelName="Qwen3-4B"
        commands={COMMANDS}
        disabledCommands={["model"]}
        onCommand={onCommand}
      />
    );
    openPanel();
    const model = screen.getByRole("button", { name: /^Model/ });
    expect(model.disabled).toBe(true);
  });

  it("shows the red attention state for an orphaned conversation, its Model button still working", () => {
    const onCommand = vi.fn();
    render(<ConversationHeader modelName="" attention commands={COMMANDS} onCommand={onCommand} />);
    expect(screen.getByRole("alert").textContent).toBe("Please select a model");
    openPanel();
    fireEvent.click(screen.getByRole("button", { name: /^Model/ }));
    expect(onCommand).toHaveBeenCalledWith({ name: "model", arg: null });
  });

  it("renders what the page adds to the panel after the command buttons", () => {
    render(
      <ConversationHeader modelName="Qwen3-4B" commands={COMMANDS} onCommand={vi.fn()}>
        <button>extra action</button>
      </ConversationHeader>
    );
    openPanel();
    expect(screen.getByText("extra action")).toBeTruthy();
  });

  it("shows a failed save", () => {
    render(
      <ConversationHeader modelName="Qwen3-4B" commands={COMMANDS} onCommand={vi.fn()} saveFailed />
    );
    expect(screen.getByRole("alert").textContent).toBe("Your change could not be saved.");
  });
});
