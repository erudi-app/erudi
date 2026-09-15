// Human-readable memory size for the amber memory warning (1.1.2).
//
// Binary units (the backend's accounting is KV-cache bytes): one decimal above
// 1 GiB where precision matters to the user, plain integers below. Returns
// null for anything that is not a non-negative finite number so callers can
// skip the size rather than render garbage.
const KIB = 1024;
const MIB = 1024 ** 2;
const GIB = 1024 ** 3;

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return null;
  }
  if (bytes >= GIB) {
    return `${(Math.floor((bytes / GIB) * 10) / 10).toFixed(1)} GB`;
  }
  if (bytes >= MIB) {
    return `${Math.round(bytes / MIB)} MB`;
  }
  if (bytes === 0) {
    return "0 KB";
  }
  return `${Math.max(1, Math.round(bytes / KIB))} KB`;
}
