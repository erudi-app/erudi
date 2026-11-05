/**
 * The slash commands of the chat composer and the conversation header (#585).
 *
 * A command's `name` is a fixed English identifier, typed as `/<name>` in
 * every interface language and never translated; its label and one-line
 * description are `chat:commands.<name>.label` / `.description`. `args` lists
 * the only words a command accepts after its name (none, or `global` for
 * `/config`), and `scopes` the pages that offer it: a conversation, or the
 * new-conversation page where the commands edit the draft it is created from.
 *
 * The grammar is exact: a message runs a command only when it is `/<name>` or
 * `/<name> <arg>` with an accepted argument (surrounding whitespace aside).
 * Anything else is an ordinary message for the model: an unknown name, an
 * argument the command does not take, or text before the command. On the
 * new-conversation page `/compact` and `/context` (which need a conversation)
 * are not offered; typed exactly there, they send nothing either.
 *
 * Two orders, both explicit: the registry order is the composer's drop-up
 * (/config, /context, /compact, /effort, /model), and `headerOrder` places
 * each command among the header panel's buttons (Model, Config, Effort,
 * Compact, Context). A later command is one more entry, inserted where the
 * drop-up lists it, with its `headerOrder`.
 */

export const COMMAND_SCOPES = Object.freeze({
  CONVERSATION: "conversation",
  DRAFT: "draft",
});

const BOTH = [COMMAND_SCOPES.CONVERSATION, COMMAND_SCOPES.DRAFT];
const CONVERSATION_ONLY = [COMMAND_SCOPES.CONVERSATION];

export const SLASH_COMMANDS = Object.freeze([
  Object.freeze({ name: "config", args: ["global"], scopes: BOTH, headerOrder: 2 }),
  Object.freeze({ name: "context", args: [], scopes: CONVERSATION_ONLY, headerOrder: 5 }),
  Object.freeze({ name: "compact", args: [], scopes: CONVERSATION_ONLY, headerOrder: 4 }),
  Object.freeze({ name: "effort", args: [], scopes: BOTH, headerOrder: 3 }),
  Object.freeze({ name: "model", args: [], scopes: BOTH, headerOrder: 1 }),
]);

/** The commands a page offers, in registry order. */
export function commandsFor(scope, registry = SLASH_COMMANDS) {
  return registry.filter((command) => command.scopes.includes(scope));
}

/** The commands a page does not offer, in registry order. */
export function commandsOutside(scope, registry = SLASH_COMMANDS) {
  return registry.filter((command) => !command.scopes.includes(scope));
}

/** `commands` in the header panel's order (a new array). */
export function headerCommands(commands) {
  return [...commands].sort((a, b) => a.headerOrder - b.headerOrder);
}

/**
 * The command `text` runs among `commands`, as `{ name, arg }` (`arg` null
 * when none was given), or null when `text` is an ordinary message.
 */
export function parseSlashCommand(text, commands) {
  const tokens = (text || "").trim().split(/\s+/);
  if (tokens.length > 2 || !tokens[0].startsWith("/")) {
    return null;
  }
  const command = commands.find((c) => `/${c.name}` === tokens[0]);
  if (!command) {
    return null;
  }
  if (tokens.length === 1) {
    return { name: command.name, arg: null };
  }
  return command.args.includes(tokens[1]) ? { name: command.name, arg: tokens[1] } : null;
}

/**
 * The commands the drop-up lists while the composer holds `text`: those whose
 * `/<name>` starts with it, as long as `text` is a slash followed by no
 * whitespace (a space after a complete name closes the list).
 */
export function matchCommands(text, commands) {
  if (!/^\/\S*$/.test(text || "")) {
    return [];
  }
  return commands.filter((command) => `/${command.name}`.startsWith(text));
}
