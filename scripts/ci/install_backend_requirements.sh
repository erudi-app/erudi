#!/usr/bin/env bash
# Install a backend requirements entrypoint the way the shipped build needs it.
#
# Usage: scripts/ci/install_backend_requirements.sh <requirements file>
#        scripts/ci/install_backend_requirements.sh backend/requirements/entrypoints/prod/mac-silicon-prod.txt
#
# app-build-smoke.yml and release.yml both call this, so the smoke gate installs
# exactly what a release installs.
#
# Off macOS it is a plain `pip install -r`. On macOS the runner is newer than the
# app's floor (mac.minimumSystemVersion in frontend/electron-builder.yml), and pip
# prefers the wheels built for the runner's own macOS, which the frozen backend
# then needs at import time. So it downloads the wheels built for the floor
# (`--only-binary=:all: --platform macosx_<floor>_arm64`) into a wheelhouse,
# installs from it alone, and exports MACOS_FLOOR to $GITHUB_ENV for
# scripts/ci/check_macos_floor.sh.
#
# Restricting pip to the floor's wheels can make it pick an older version of a
# package whose newest release has no wheel for the floor; most transitive
# dependencies are not pinned, so that would ship a version the native install of
# backend-ci never tested. The script resolves the same file natively
# (`pip install --dry-run --report`) and fails when any installed version differs,
# naming the packages: pin them, or raise the floor.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <requirements file>" >&2
  exit 2
fi
reqs=$1

python -m pip install --upgrade pip wheel setuptools

if [ "$(uname -s)" != "Darwin" ]; then
  pip install -r "$reqs"
  exit 0
fi

# Read minimumSystemVersion from the top-level `mac:` block only (dmg, mas and
# pkg blocks may carry their own), accepting a quoted or bare value and a
# trailing comment.
floors=$(awk '
  /^[^[:space:]#]/ { in_mac = ($0 ~ /^mac:/) }
  in_mac && /^[[:space:]]+minimumSystemVersion:/ {
    v = $0
    sub(/^[^:]*:[[:space:]]*/, "", v)
    sub(/[[:space:]]*#.*$/, "", v)
    gsub(/["\047]/, "", v)
    print v
  }
' frontend/electron-builder.yml)
if [ "$(printf '%s\n' "$floors" | grep -c .)" -ne 1 ]; then
  echo "::error::Expected one mac.minimumSystemVersion in frontend/electron-builder.yml, found: '${floors}'."
  exit 1
fi
floor=$floors
if ! [[ "$floor" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "::error::mac.minimumSystemVersion '$floor' is not a version like 14.0."
  exit 1
fi
# pip's platform tag wants MAJOR_MINOR: 14 and 14.0 both become 14_0.
major=${floor%%.*}
minor=0
if [ "$floor" != "$major" ]; then minor=${floor#*.}; fi
platform="macosx_${major}_${minor}_arm64"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "MACOS_FLOOR=$floor" >> "$GITHUB_ENV"
fi
echo "macOS floor $floor: installing the $platform wheels."

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
pip download -r "$reqs" -d "$work/wheelhouse" --only-binary=:all: --platform "$platform"
pip install --dry-run --ignore-installed --quiet --report "$work/native.json" -r "$reqs"
pip install --no-index --find-links "$work/wheelhouse" -r "$reqs"
pip list --format=freeze > "$work/installed.txt"

# Every package the native resolution picked must be installed at that version.
python - "$work/native.json" "$work/installed.txt" <<'PY'
import json
import re
import sys

def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()

with open(sys.argv[1], encoding="utf-8") as f:
    native = {
        canon(item["metadata"]["name"]): item["metadata"]["version"]
        for item in json.load(f)["install"]
    }
installed = {}
with open(sys.argv[2], encoding="utf-8") as f:
    for line in f:
        name, sep, version = line.strip().partition("==")
        if sep:
            installed[canon(name)] = version
drift = sorted(
    f"{name}: {installed.get(name, 'missing')} installed, {version} natively"
    for name, version in native.items()
    if installed.get(name) != version
)
if drift:
    print(
        "::error::The floor's wheels resolve to other versions than a native install; "
        "pin these or raise mac.minimumSystemVersion:"
    )
    for line in drift:
        print("  " + line)
    sys.exit(1)
print(f"{len(native)} packages installed at the versions a native install resolves.")
PY
