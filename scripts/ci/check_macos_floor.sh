#!/usr/bin/env bash
# Fail when a Mach-O file under <dir> needs a newer macOS than <floor>.
#
# Usage: scripts/ci/check_macos_floor.sh <dir> <floor>
#        scripts/ci/check_macos_floor.sh backend/dist 14.0
#
# The shipped app declares its floor in frontend/electron-builder.yml
# (mac.minimumSystemVersion). The frozen backend inherits the floor of every
# wheel pip picked on the build runner, and pip prefers the wheel built for the
# runner's own macOS: on a newer runner it takes a newer wheel (mlx, mlx-metal
# and orjson publish one per macOS release), and the backend then dies at
# import time on the older systems the installer still accepts. This walks every
# Mach-O file (`file` says "Mach-O"; universal binaries included, each slice
# read), reads its minimum macOS from `otool -l` (LC_BUILD_VERSION `minos`, or
# LC_VERSION_MIN_MACOSX `version` for older binaries), and lists every file
# whose minimum is above <floor>.
#
# Exit codes: 0 every file is at or below the floor; 1 at least one is above it;
# 2 usage error, unreadable file, or no Mach-O file found (an empty walk proves
# nothing).
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <dir> <floor>" >&2
  exit 2
fi
dir=$1
floor=$2
if [ ! -d "$dir" ]; then
  echo "error: $dir is not a directory" >&2
  exit 2
fi
if ! [[ "$floor" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
  echo "error: floor '$floor' is not a version like 14.0" >&2
  exit 2
fi

# Succeeds when dotted version $1 is above $2; missing components count as 0,
# so 14 and 14.0 are equal.
version_gt() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    na = split(a, x, "."); nb = split(b, y, ".")
    n = na > nb ? na : nb
    for (i = 1; i <= n; i++) {
      if ((x[i] + 0) > (y[i] + 0)) exit 0
      if ((x[i] + 0) < (y[i] + 0)) exit 1
    }
    exit 1
  }'
}

checked=0
unversioned=0
too_new=()
highest=0
# `file -0` prints "<path>\0: <description>\n". For a universal binary it then
# prints one "<path> (for architecture <arch>): ..." line per slice, which this
# reads in front of the next path: the path is the last line of what precedes
# the NUL.
while IFS= read -r -d '' chunk && IFS= read -r description; do
  path=${chunk##*$'\n'}
  case "$description" in
    *Mach-O*) ;;
    *) continue ;;
  esac
  checked=$((checked + 1))
  load_commands=$(otool -arch all -l "$path") || {
    echo "error: otool could not read $path" >&2
    exit 2
  }
  # One minimum per slice (one slice per architecture); the file needs the greatest.
  minos=$(printf '%s\n' "$load_commands" | awk '
    $2 == "LC_BUILD_VERSION" || $2 == "LC_VERSION_MIN_MACOSX" { want = 1; next }
    want && ($1 == "minos" || $1 == "version") { print $2; want = 0 }
  ')
  if [ -z "$minos" ]; then
    unversioned=$((unversioned + 1))
    continue
  fi
  file_min=0
  for v in $minos; do
    if version_gt "$v" "$file_min"; then file_min=$v; fi
  done
  if version_gt "$file_min" "$highest"; then highest=$file_min; fi
  if version_gt "$file_min" "$floor"; then
    too_new+=("$file_min  $path")
  fi
done < <(find "$dir" -type f -exec file -0 {} +)

if [ "$checked" -eq 0 ]; then
  echo "error: no Mach-O file under $dir; nothing was checked" >&2
  exit 2
fi

echo "Checked $checked Mach-O files under $dir (highest minimum: macOS $highest; $unversioned carry none)."
if [ "${#too_new[@]}" -gt 0 ]; then
  echo "error: ${#too_new[@]} files need a newer macOS than the floor $floor:" >&2
  printf '  %s\n' "${too_new[@]}" >&2
  exit 1
fi
echo "Every file runs on macOS $floor."
