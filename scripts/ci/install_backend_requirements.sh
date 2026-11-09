#!/usr/bin/env bash
# Install what a release build freezes, from the lock, into a fresh venv.
#
# Usage: scripts/ci/install_backend_requirements.sh <prod requirements file>
#        scripts/ci/install_backend_requirements.sh backend/requirements/entrypoints/prod/mac-silicon-prod.txt
#
# app-build-smoke.yml and release.yml both call this, so the smoke gate installs
# exactly what a release installs.
#
# It creates a venv under $RUNNER_TEMP (a temporary directory off CI): the
# runner's Python carries packages of its own (pipx, argcomplete and userpath on
# Windows; click, colorama, filelock, packaging and platformdirs there too) that
# would otherwise stand in for locked ones or reach PyInstaller. Into it, it
# installs the entrypoint's lock, backend/requirements/lock/<same name>, alone
# and in hash-checking mode (`--require-hashes --no-deps`): the lock carries
# the index options, every package of the entrypoint and of
# backend/requirements/meta/build.txt (PyInstaller), and the sha256 of one file
# per package, so the build gets those files and nothing published since. It
# then runs `pip check` (the dependencies the lock was resolved with hold here)
# and scripts/ci/check_backend_lock.py, which fails unless the venv holds exactly
# the lock. That the lock still matches its requirement files is checked by
# backend-ci on every pull request (backend/tests/test_requirements_lock.py and
# `check_backend_lock.py resolution`).
#
# The venv's bin directory (Scripts on Windows) is appended to $GITHUB_PATH, so
# the workflow's later steps (pyinstaller, python) run from it.
#
# On macOS the lock holds the wheels built for the app's floor
# (mac.minimumSystemVersion in frontend/electron-builder.yml), not the newer ones
# pip would prefer on the runner, and the hashes make pip take them. The floor is
# exported as MACOS_FLOOR to $GITHUB_ENV for scripts/ci/check_macos_floor.sh.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <prod requirements file>" >&2
  exit 2
fi
lock="backend/requirements/lock/$(basename "$1")"
if [ ! -f "$lock" ]; then
  echo "::error::No lock for $1: expected $lock. Generate it with scripts/dev/lock_backend_requirements.py."
  exit 1
fi

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    root=$(cygpath -u "${RUNNER_TEMP:-$(mktemp -d)}")
    bin=Scripts
    ;;
  *)
    root=${RUNNER_TEMP:-$(mktemp -d)}
    bin=bin
    ;;
esac
venv="$root/backend-venv"
rm -rf "$venv"
python -m venv "$venv"
venv_python="$venv/$bin/python"

"$venv_python" -m pip install --disable-pip-version-check \
  --require-hashes --no-deps --only-binary=:all: -r "$lock"
"$venv_python" -m pip check
"$venv_python" scripts/ci/check_backend_lock.py installed "$lock"

if [ -n "${GITHUB_PATH:-}" ]; then
  if [ "$bin" = Scripts ]; then
    cygpath -w "$venv/$bin" >> "$GITHUB_PATH"
  else
    echo "$venv/$bin" >> "$GITHUB_PATH"
  fi
fi
echo "Backend venv: $venv"

if [ "$(uname -s)" != "Darwin" ]; then
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
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "MACOS_FLOOR=$floors" >> "$GITHUB_ENV"
fi
