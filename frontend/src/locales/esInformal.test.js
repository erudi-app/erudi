import { describe, it, expect } from "vitest";
import { resources } from "../i18n/resources";

// The Spanish interface addresses the user as "tú", like the rest of its copy
// ("Vuelve a hacer tu pregunta"). The formal "usted" forms that once crept into
// the memory notices must not come back.

const USTED = /\b(su equipo|su tarjeta|Inicie|envíe|cambie|libere usted)\b/;

function leaves(obj, prefix = "") {
  return Object.entries(obj).flatMap(([key, value]) =>
    value !== null && typeof value === "object"
      ? leaves(value, `${prefix}${key}.`)
      : [[`${prefix}${key}`, String(value)]]
  );
}

describe("Spanish chat copy", () => {
  it("uses tú, never the usted forms", () => {
    const formal = leaves(resources.es.chat)
      .filter(([, value]) => USTED.test(value))
      .map(([key]) => key);
    expect(formal).toEqual([]);
  });
});
