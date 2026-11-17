import i18n from "../i18n";
import { formatDate } from "../i18n/format";

/**
 * "Published March 2026" in the interface language, or null when the model has
 * no known publication date (#561): a card says nothing rather than a wrong
 * date. `isoDate` is the backend's `published_at` (ISO-8601 or null). The month
 * is read in UTC so a release on the 1st or the 31st never lands in the
 * neighbouring month for a user in another time zone.
 */
export function publishedLabel(isoDate) {
  if (!isoDate) return null;
  const date = formatDate(isoDate, { year: "numeric", month: "long", timeZone: "UTC" });
  if (!date) return null;
  return i18n.t("models:card.published", { date });
}
