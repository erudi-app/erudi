#!/usr/bin/env bash
# check_mac_app_signature.sh
#
# Usage:
#   check_mac_app_signature.sh <Erudi.app>              # local and smoke builds
#   check_mac_app_signature.sh --release <Erudi.app>    # signed and notarized release
#
# Verify only: nothing is ever re-signed here.
#
# Both modes require `codesign --verify --deep --strict` to pass on the whole
# bundle. An app whose signature covers the main executable only fails it ("code
# has no resources but signature indicates they must be present") and macOS 26
# kills it at launch (#690). Builds without a signing identity are signed ad hoc
# by electron-builder (`-c.mac.identity=-`), which signs every file.
#
# Default mode also requires the hardened runtime flag in `codesign -dvv`
# (flags=0x10000(runtime)), so a regression of the hardened runtime is caught on
# builds that carry no certificate.
#
# --release also requires the Developer ID Application authority, the team
# identifier of the release certificate and a valid notarization ticket
# (`xcrun stapler validate`).

set -euo pipefail

RELEASE_TEAM_ID="FAF8JCAT86"

RELEASE=0
if [ "${1:-}" = "--release" ]; then
    RELEASE=1
    shift
fi
APP="${1:-}"

fail() { echo "error: $*" >&2; exit 1; }

if [ -z "$APP" ] || [ ! -d "$APP" ]; then
    echo "error: app bundle not found: '$APP'" >&2
    exit 2
fi

codesign --verify --deep --strict --verbose=2 "$APP" >&2 \
    || fail "codesign --verify --deep --strict failed on $APP"

# `codesign -dvv` writes to stderr.
DETAILS="$(codesign -dvv "$APP" 2>&1)" || fail "codesign -dvv failed on $APP"

if [ "$RELEASE" -eq 1 ]; then
    grep -q '^Authority=Developer ID Application:' <<<"$DETAILS" \
        || fail "no Developer ID Application authority on $APP"
    grep -q "^TeamIdentifier=$RELEASE_TEAM_ID\$" <<<"$DETAILS" \
        || fail "TeamIdentifier is not $RELEASE_TEAM_ID on $APP"
    xcrun stapler validate "$APP" >&2 || fail "no valid notarization ticket stapled to $APP"
else
    grep -Eq 'flags=0x[0-9a-f]+\([^)]*runtime' <<<"$DETAILS" \
        || fail "the hardened runtime flag is not set on $APP"
fi
echo "verified"
