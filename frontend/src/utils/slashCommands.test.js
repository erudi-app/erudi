import { describe, it, expect } from "vitest";
import {
  COMMAND_SCOPES,
  SLASH_COMMANDS,
  commandsFor,
  commandsOutside,
  headerCommands,
  matchCommands,
  parseSlashCommand,
} from "./slashCommands";

// The slash-command registry (#585): fixed English identifiers, an exact
// grammar per command, and the page each command is offered on.

const ALL = commandsFor(COMMAND_SCOPES.CONVERSATION);

describe("the registry", () => {
  it("offers the five commands in a conversation, and three on the new-conversation page", () => {
    expect(SLASH_COMMANDS.map((c) => c.name)).toEqual([
      "config",
      "context",
      "compact",
      "effort",
      "model",
    ]);
    expect(commandsFor(COMMAND_SCOPES.CONVERSATION).map((c) => c.name)).toEqual([
      "config",
      "context",
      "compact",
      "effort",
      "model",
    ]);
    expect(commandsFor(COMMAND_SCOPES.DRAFT).map((c) => c.name)).toEqual([
      "config",
      "effort",
      "model",
    ]);
  });

  it("keeps two explicit orders: the drop-up's (the registry's) and the header panel's", () => {
    expect(matchCommands("/", ALL).map((c) => c.name)).toEqual([
      "config",
      "context",
      "compact",
      "effort",
      "model",
    ]);
    expect(headerCommands(ALL).map((c) => c.name)).toEqual([
      "model",
      "config",
      "effort",
      "compact",
      "context",
    ]);
    // headerCommands never reorders the list it is given.
    expect(ALL.map((c) => c.name)[0]).toBe("config");
  });

  it("names the commands a page does not offer: /compact and /context before a conversation exists", () => {
    expect(commandsOutside(COMMAND_SCOPES.DRAFT).map((c) => c.name)).toEqual([
      "context",
      "compact",
    ]);
    expect(commandsOutside(COMMAND_SCOPES.CONVERSATION)).toEqual([]);
  });

  it("filters an entry out of a page it is not scoped to", () => {
    const registry = [
      { name: "config", args: ["global"], scopes: [COMMAND_SCOPES.CONVERSATION] },
      { name: "effort", args: [], scopes: [COMMAND_SCOPES.CONVERSATION, COMMAND_SCOPES.DRAFT] },
    ];
    expect(commandsFor(COMMAND_SCOPES.DRAFT, registry).map((c) => c.name)).toEqual(["effort"]);
    expect(commandsOutside(COMMAND_SCOPES.DRAFT, registry).map((c) => c.name)).toEqual(["config"]);
  });
});

describe("parseSlashCommand: exact match only", () => {
  it.each([
    ["/config", { name: "config", arg: null }],
    ["/config global", { name: "config", arg: "global" }],
    ["/effort", { name: "effort", arg: null }],
    ["/model", { name: "model", arg: null }],
    ["/compact", { name: "compact", arg: null }],
    ["/context", { name: "context", arg: null }],
    ["  /model  ", { name: "model", arg: null }],
  ])("%j runs a command", (text, expected) => {
    expect(parseSlashCommand(text, ALL)).toEqual(expected);
  });

  it.each([
    ["please /config this"],
    ["/foo"],
    ["/config everything"],
    ["/config global now"],
    ["/effort high"],
    ["/compact now"],
    ["/context full"],
    ["please /compact this for me"],
    ["/model qwen"],
    ["/Config"],
    ["config"],
    ["/"],
    [""],
    ["/conf"],
  ])("%j is a normal message", (text) => {
    expect(parseSlashCommand(text, ALL)).toBeNull();
  });

  it("does not run a command the page does not offer", () => {
    const onlyEffort = ALL.filter((c) => c.name === "effort");
    expect(parseSlashCommand("/config", onlyEffort)).toBeNull();
  });
});

describe("matchCommands: the drop-up list", () => {
  it("lists every command for a bare slash", () => {
    expect(matchCommands("/", ALL).map((c) => c.name)).toEqual([
      "config",
      "context",
      "compact",
      "effort",
      "model",
    ]);
  });

  it("narrows by prefix", () => {
    expect(matchCommands("/con", ALL).map((c) => c.name)).toEqual(["config", "context"]);
    expect(matchCommands("/com", ALL).map((c) => c.name)).toEqual(["compact"]);
    expect(matchCommands("/m", ALL).map((c) => c.name)).toEqual(["model"]);
  });

  it("keeps a complete name listed until a space follows it", () => {
    expect(matchCommands("/config", ALL).map((c) => c.name)).toEqual(["config"]);
    expect(matchCommands("/config ", ALL)).toEqual([]);
    expect(matchCommands("/config global", ALL)).toEqual([]);
  });

  it("lists nothing for text that is not a command prefix", () => {
    expect(matchCommands("", ALL)).toEqual([]);
    expect(matchCommands("hello /config", ALL)).toEqual([]);
    expect(matchCommands("/zzz", ALL)).toEqual([]);
    expect(matchCommands(" /config", ALL)).toEqual([]);
  });
});
