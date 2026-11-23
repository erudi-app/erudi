// Human-readable memory size for the amber memory warning (1.1.2).
//
// Binary units (the backend's accounting is KV-cache bytes): one decimal above
// 1 GiB where precision matters to the user, plain integers below. The number
// follows the interface language's separators and the unit its `common:units`
// label ("4,0 Go"). Returns null for anything that is not a non-negative
// finite number so callers can skip the size rather than render garbage.
import i18n from "../i18n";
import { formatNumber } from "../i18n/format";

const KIB = 1024;
const MIB = 1024 ** 2;
const GIB = 1024 ** 3;

const withUnit = (number, unit) => `${number} ${i18n.t(`common:units.${unit}`)}`;

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return null;
  }
  if (bytes >= GIB) {
    const tenths = Math.floor((bytes / GIB) * 10) / 10;
    return withUnit(
      formatNumber(tenths, { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
      "gb"
    );
  }
  if (bytes >= MIB) {
    return withUnit(formatNumber(Math.round(bytes / MIB)), "mb");
  }
  if (bytes === 0) {
    return withUnit(formatNumber(0), "kb");
  }
  return withUnit(formatNumber(Math.max(1, Math.round(bytes / KIB))), "kb");
}
