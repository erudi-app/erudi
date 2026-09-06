// @vitest-environment jsdom
/**
 * The Diagnostics anchor under HashRouter.
 *
 * The app routes with `HashRouter`, so the whole route already lives in the
 * URL fragment. `#/erudi/settings#diagnostics` therefore has two `#`, and the
 * browser will not scroll to the second one on its own — SettingsPage does the
 * scroll. What this test pins is the half that must hold for that to be
 * possible: react-router still parses the path into a pathname and a hash.
 */
import { describe, it, expect } from "vitest";
import { parsePath } from "react-router-dom";

import { DIAGNOSTICS_PATH, SETTINGS_PATH } from "./routes";

describe("DIAGNOSTICS_PATH", () => {
  it("points at the settings route with the panel's anchor", () => {
    expect(DIAGNOSTICS_PATH).toBe("/erudi/settings#diagnostics");
  });

  it("parses into the settings pathname and the panel's hash", () => {
    const parsed = parsePath(DIAGNOSTICS_PATH);
    expect(parsed.pathname).toBe(SETTINGS_PATH);
    expect(parsed.hash).toBe("#diagnostics");
  });
});
