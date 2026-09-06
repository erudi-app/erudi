/**
 * Static translation bundle (#385).
 *
 * Every namespace of every language is imported here so webpack inlines the
 * catalogs into the renderer bundle — the packaged app runs under a strict
 * CSP with no remote loading, and the boot screen must already speak the
 * user's language before the backend is reachable. Adding a namespace means
 * adding it to NAMESPACES and one import per language; `locales.test.js`
 * fails if a language is missing a file or a key.
 */
import enArena from "../locales/en/arena.json";
import enChat from "../locales/en/chat.json";
import enCommon from "../locales/en/common.json";
import enDiagnostics from "../locales/en/diagnostics.json";
import enDownloads from "../locales/en/downloads.json";
import enErrors from "../locales/en/errors.json";
import enKnowledgeBase from "../locales/en/knowledgeBase.json";
import enLanding from "../locales/en/landing.json";
import enMain from "../locales/en/main.json";
import enModels from "../locales/en/models.json";
import enSettings from "../locales/en/settings.json";

import frArena from "../locales/fr/arena.json";
import frChat from "../locales/fr/chat.json";
import frCommon from "../locales/fr/common.json";
import frDiagnostics from "../locales/fr/diagnostics.json";
import frDownloads from "../locales/fr/downloads.json";
import frErrors from "../locales/fr/errors.json";
import frKnowledgeBase from "../locales/fr/knowledgeBase.json";
import frLanding from "../locales/fr/landing.json";
import frMain from "../locales/fr/main.json";
import frModels from "../locales/fr/models.json";
import frSettings from "../locales/fr/settings.json";

import esArena from "../locales/es/arena.json";
import esChat from "../locales/es/chat.json";
import esCommon from "../locales/es/common.json";
import esDiagnostics from "../locales/es/diagnostics.json";
import esDownloads from "../locales/es/downloads.json";
import esErrors from "../locales/es/errors.json";
import esKnowledgeBase from "../locales/es/knowledgeBase.json";
import esLanding from "../locales/es/landing.json";
import esMain from "../locales/es/main.json";
import esModels from "../locales/es/models.json";
import esSettings from "../locales/es/settings.json";

import zhArena from "../locales/zh/arena.json";
import zhChat from "../locales/zh/chat.json";
import zhCommon from "../locales/zh/common.json";
import zhDiagnostics from "../locales/zh/diagnostics.json";
import zhDownloads from "../locales/zh/downloads.json";
import zhErrors from "../locales/zh/errors.json";
import zhKnowledgeBase from "../locales/zh/knowledgeBase.json";
import zhLanding from "../locales/zh/landing.json";
import zhMain from "../locales/zh/main.json";
import zhModels from "../locales/zh/models.json";
import zhSettings from "../locales/zh/settings.json";

export const NAMESPACES = [
  "common",
  "settings",
  "models",
  "downloads",
  "chat",
  "arena",
  "knowledgeBase",
  "landing",
  "errors",
  "main",
  "diagnostics",
];

export const DEFAULT_NAMESPACE = "common";

export const resources = {
  en: {
    arena: enArena,
    chat: enChat,
    common: enCommon,
    diagnostics: enDiagnostics,
    downloads: enDownloads,
    errors: enErrors,
    knowledgeBase: enKnowledgeBase,
    landing: enLanding,
    main: enMain,
    models: enModels,
    settings: enSettings,
  },
  fr: {
    arena: frArena,
    chat: frChat,
    common: frCommon,
    diagnostics: frDiagnostics,
    downloads: frDownloads,
    errors: frErrors,
    knowledgeBase: frKnowledgeBase,
    landing: frLanding,
    main: frMain,
    models: frModels,
    settings: frSettings,
  },
  es: {
    arena: esArena,
    chat: esChat,
    common: esCommon,
    diagnostics: esDiagnostics,
    downloads: esDownloads,
    errors: esErrors,
    knowledgeBase: esKnowledgeBase,
    landing: esLanding,
    main: esMain,
    models: esModels,
    settings: esSettings,
  },
  zh: {
    arena: zhArena,
    chat: zhChat,
    common: zhCommon,
    diagnostics: zhDiagnostics,
    downloads: zhDownloads,
    errors: zhErrors,
    knowledgeBase: zhKnowledgeBase,
    landing: zhLanding,
    main: zhMain,
    models: zhModels,
    settings: zhSettings,
  },
};
