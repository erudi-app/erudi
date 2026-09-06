/**
 * The Diagnostics route.
 *
 * Diagnostics is a page of its own, reached from the bug button at the bottom
 * of the sidebar rail. It is not a section of Settings, so its path carries no
 * fragment: under `HashRouter` the whole route already lives in the URL hash,
 * and a second `#` would be something the browser cannot follow on its own.
 */
import { describe, it, expect } from "vitest";
import { parsePath } from "react-router-dom";

import { DIAGNOSTICS_PATH, SETTINGS_PATH } from "./routes";

describe("DIAGNOSTICS_PATH", () => {
  it("is a route of its own", () => {
    expect(DIAGNOSTICS_PATH).toBe("/erudi/diagnostics");
  });

  it("does not live under the settings route", () => {
    const parsed = parsePath(DIAGNOSTICS_PATH);
    expect(parsed.pathname).not.toBe(SETTINGS_PATH);
    expect(parsed.hash).toBeFalsy();
  });
});
