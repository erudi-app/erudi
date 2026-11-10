// Single source of truth for in-app route paths, so a path is never hand-typed
// (and mistyped) at a call site. The conversation route is PLURAL — a singular
// "/erudi/conversation/:id" navigation silently hit the catch-all redirect (#133).

/** Path to a single conversation by id. */
export const conversationPath = (id) => `/erudi/conversations/${id}`;

/** Path to the app settings page. */
export const SETTINGS_PATH = "/erudi/settings";

/** Path to the diagnostics page (the bug button at the bottom of the rail). */
export const DIAGNOSTICS_PATH = "/erudi/diagnostics";

/** Path to the Models page: installed models, the catalog, and re-binding an orphaned assistant. */
export const MODELS_PATH = "/erudi/models";
