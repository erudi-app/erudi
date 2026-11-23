/**
 * Locale-aware formatting routed through the active app language (#385).
 *
 * Components must not build their own `Intl.*` formatters: import these so
 * numbers, sizes, percentages and dates follow the language picked in
 * Settings. The unit labels come from the `common` namespace so "GB" reads
 * "Go" in French.
 */
import i18n from "./index";

const LOCALE_BY_LANGUAGE = {
  en: "en-US",
  fr: "fr-FR",
  es: "es-ES",
  zh: "zh-CN",
};

/** The Intl locale tag for the active language (English when unknown). */
export function currentLocale() {
  return LOCALE_BY_LANGUAGE[i18n.language] || LOCALE_BY_LANGUAGE.en;
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatNumber(value, options = {}) {
  if (!isFiniteNumber(value)) return "";
  return new Intl.NumberFormat(currentLocale(), options).format(value);
}

/** "42.3%" / "42,3 %" — one decimal by default; `value` is 0-100. */
export function formatPercent(value, { maximumFractionDigits = 1 } = {}) {
  if (!isFiniteNumber(value)) return "";
  return new Intl.NumberFormat(currentLocale(), {
    style: "percent",
    maximumFractionDigits,
  }).format(value / 100);
}

const SIZE_UNITS = ["b", "kb", "mb", "gb", "tb"];

/**
 * Bytes → "1.5 GB" (units translated). Decimal (1 GB = 1e9 bytes) for every
 * file and transfer, like the model card, Hugging Face and Finder; memory
 * (RAM/VRAM, KV cache) stays binary and is formatted elsewhere.
 */
export function formatFileSize(bytes, { maximumFractionDigits = 1 } = {}) {
  if (!isFiniteNumber(bytes) || bytes < 0) return "";
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1000 && unitIndex < SIZE_UNITS.length - 1) {
    value /= 1000;
    unitIndex += 1;
  }
  const number = formatNumber(value, { maximumFractionDigits });
  return `${number} ${i18n.t(`common:units.${SIZE_UNITS[unitIndex]}`)}`;
}

/** A value already expressed in GB (the catalog's unit) → "1.5 GB" / "1,5 Go". */
export function formatGigabytes(gigabytes, { maximumFractionDigits = 1 } = {}) {
  if (!isFiniteNumber(gigabytes)) return "";
  return `${formatNumber(gigabytes, { maximumFractionDigits })} ${i18n.t("common:units.gb")}`;
}

/**
 * Join items as a list in the app language ("a, b, and c", "a, b et c",
 * "a、b和c"). The conjunction form: the unit form has no separator at all in
 * Chinese.
 */
export function formatList(items) {
  return new Intl.ListFormat(currentLocale(), { type: "conjunction" }).format(items);
}

export function formatDate(value, options = { dateStyle: "medium" }) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(currentLocale(), options).format(date);
}
