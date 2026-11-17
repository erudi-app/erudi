#!/usr/bin/env bash
# check_mac_app_signature.sh
#
# Usage:
#   check_mac_app_signature.sh <Erudi.app>                 # local / smoke builds
#   check_mac_app_signature.sh --verify-only <Erudi.app>   # release builds
#
# Fails unless `codesign --verify --deep --strict` passes on the whole bundle.
#
# An app with no signing identity (CSC_IDENTITY_AUTO_DISCOVERY=false, no
# certificate in the keychain) leaves electron-builder with an ad hoc signature
# on the main executable only. `codesign --verify --deep --strict` rejects it
# ("code has no resources but signature indicates they must be present") and
# macOS 26 kills it at launch (exit 137). Without --verify-only, an app that
# carries no signing authority is therefore re-signed ad hoc as a whole
# (`codesign --force --deep --sign -`) before the check. An app signed with an
# identity is never re-signed, and --verify-only never re-signs anything.
#
# The last line of stdout is "adhoc-resigned" or "verified".

set -euo pipefail

VERIFY_ONLY=0
if [ "${1:-}" = "--verify-only" ]; then
    VERIFY_ONLY=1
    shift
fi
APP="${1:-}"

if [ -z "$APP" ] || [ ! -d "$APP" ]; then
    echo "error: app bundle not found: '$APP'" >&2
    exit 2
fi

RESULT="verified"
if [ "$VERIFY_ONLY" -eq 0 ]; then
    # `codesign -dvv` writes to stderr. A missing signature makes it exit
    # non-zero, which also means "no authority": both lead to the ad hoc re-sign.
    DETAILS="$(codesign -dvv "$APP" 2>&1 || true)"
    if ! printf '%s\n' "$DETAILS" | grep -q '^Authority='; then
        echo "App carries no signing authority: re-signing it ad hoc as a whole." >&2
        codesign --force --deep --sign - "$APP" >&2
        RESULT="adhoc-resigned"
    fi
fi

if ! codesign --verify --deep --strict --verbose=2 "$APP" >&2; then
    echo "error: codesign --verify --deep --strict failed on $APP" >&2
    exit 1
fi
echo "$RESULT"
