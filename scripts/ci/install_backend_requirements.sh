#!/usr/bin/env bash
# Install a backend requirements entrypoint, and the build tool, the way the
# shipped build needs them.
#
# Usage: scripts/ci/install_backend_requirements.sh <requirements file>
#        scripts/ci/install_backend_requirements.sh backend/requirements/entrypoints/prod/mac-silicon-prod.txt
#
# app-build-smoke.yml and release.yml both call this, so the smoke gate installs
# exactly what a release installs.
#
# The entrypoint and backend/requirements/meta/build.txt (PyInstaller) are
# installed together, constrained to the entrypoint's lock,
# backend/requirements/lock/<same name> (`-c`), which pins every package of their
# resolution, so a rebuild bundles the versions the lock names and nothing that
# was published since, PyInstaller's bootloader and hooks included.
# scripts/ci/check_installed_lock.py then fails the step when the installed set
# is not exactly the lock: a requirement file changed without the locks being
# regenerated (scripts/dev/lock_backend_requirements.py).
#
# Off macOS the install is a plain `pip install -r <file> -r <build> -c <lock>`. On macOS the
# runner is newer than the app's floor (mac.minimumSystemVersion in
# frontend/electron-builder.yml), and pip prefers the wheels built for the
# runner's own macOS, which the frozen backend then needs at import time. So it
# downloads the wheels built for the floor (`--only-binary=:all: --platform
# macosx_<floor>_arm64`) into a wheelhouse, installs from it alone, and exports
# MACOS_FLOOR to $GITHUB_ENV for scripts/ci/check_macos_floor.sh. The macOS lock
# is generated for that same floor, so every locked version has a wheel there.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <requirements file>" >&2
  exit 2
fi
reqs=$1
build=backend/requirements/meta/build.txt
lock="backend/requirements/lock/$(basename "$reqs")"
if [ ! -f "$lock" ]; then
  echo "::error::No lock for $reqs: expected $lock. Generate it with scripts/dev/lock_backend_requirements.py."
  exit 1
fi

python -m pip install --upgrade pip wheel setuptools

if [ "$(uname -s)" != "Darwin" ]; then
  pip install -r "$reqs" -r "$build" -c "$lock"
  python scripts/ci/check_installed_lock.py "$lock"
  exit 0
fi

# Read minimumSystemVersion from the top-level `mac:` block only (dmg, mas and
# pkg blocks may carry their own), accepting a quoted or bare value and a
# trailing comment. scripts/dev/lock_backend_requirements.py reads it the same
# way to generate the macOS lock.
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
pip download -r "$reqs" -r "$build" -c "$lock" -d "$work/wheelhouse" \
  --only-binary=:all: --platform "$platform"
pip install --no-index --find-links "$work/wheelhouse" -r "$reqs" -r "$build" -c "$lock"
python scripts/ci/check_installed_lock.py "$lock"
