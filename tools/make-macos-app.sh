#!/bin/sh
# Builds LocalScribe.app for Apple silicon.
#
# Self-contained, carrying the Windows decision across: the app is a folder someone is handed
# and must not care what .NET the machine has.
#
#   tools/make-macos-app.sh [--install] [--sign "Developer ID Application: …"] [--notarize PROFILE]
#
# Without --sign it is signed ad-hoc: enough for personal use and for the microphone permission
# to stick to the bundle, but another Mac refuses it. To hand it to someone else, sign with a
# Developer ID and notarize. PROFILE is a notarytool keychain profile, stored once with
#   xcrun notarytool store-credentials PROFILE --apple-id … --team-id …
# The result is build/LocalScribe-<version>-mac.zip, stapled, which opens on any Apple-silicon
# Mac with no warning.
set -e

cd "$(dirname "$0")/.."

INSTALL=
IDENTITY=
PROFILE=

while [ $# -gt 0 ]; do
    case "$1" in
        --install) INSTALL=1 ;;
        --sign) IDENTITY="$2"; shift ;;
        --notarize) PROFILE="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ -n "$PROFILE" ] && [ -z "$IDENTITY" ]; then
    echo "--notarize needs --sign: Apple notarizes only Developer ID signatures." >&2
    exit 2
fi

PUBLISH=build/publish-mac
APP=build/LocalScribe.app

dotnet publish src/LocalScribe.Desktop -c Release -r osx-arm64 --self-contained -o "$PUBLISH"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp -R "$PUBLISH/". "$APP/Contents/MacOS/"
cp tools/macos/Info.plist "$APP/Contents/"

# The same drawings as the Windows .ico files, produced by the same masters:
# tools/make-icon.py --iconset and tools/make-scrb-icon.py --iconset, then iconutil.
cp tools/macos/LocalScribe.icns tools/macos/scrb.icns "$APP/Contents/Resources/"

# The whisper.net Core ML packaging defect (see Directory.Build.targets) has to be fixed in
# whatever copy actually ships, so the patch is repeated here against the published dylib.
COREML="$APP/Contents/MacOS/runtimes/coreml/macos-arm64/libwhisper.dylib"
if [ -f "$COREML" ] && ! otool -l "$COREML" | grep -q "@loader_path"; then
    install_name_tool -add_rpath @loader_path "$COREML"
fi

# This repo lives in a OneDrive folder, and the sync client stamps extended attributes on
# everything it touches — which codesign refuses as "detritus". Stripped, not worked around:
# the attributes carry nothing the app needs.
xattr -cr "$APP"

if [ -z "$IDENTITY" ]; then
    codesign --force --deep --sign - "$APP"
else
    # Inside out, every file on its own: --deep signs nested code without the hardened
    # runtime or a timestamp, and notarization rejects both omissions. Every file, not only
    # the native ones: codesign counts anything in Contents/MacOS as nested code, the managed
    # .dll files included, and refuses the bundle while one is unsigned. A non-Mach-O file
    # carries its signature in extended attributes, which ditto keeps in the shared zip.
    # This is Avalonia's documented macOS recipe.
    MAIN="$APP/Contents/MacOS/LocalScribe.Desktop"
    find "$APP/Contents/MacOS" -type f | while read -r f; do
        if [ "$f" != "$MAIN" ]; then
            codesign --force --timestamp --options runtime --sign "$IDENTITY" "$f" 2>/dev/null \
                || codesign --force --timestamp --options runtime --sign "$IDENTITY" "$f"
        fi
    done

    codesign --force --timestamp --options runtime \
        --entitlements tools/macos/LocalScribe.entitlements \
        --sign "$IDENTITY" "$APP"

    codesign --verify --strict --deep "$APP"
    echo "Signed with $IDENTITY"
fi

VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP/Contents/Info.plist")
SHARE="build/LocalScribe-$VERSION-mac.zip"

if [ -n "$PROFILE" ]; then
    SUBMIT=build/LocalScribe-notarize.zip
    rm -f "$SUBMIT"
    ditto -c -k --keepParent "$APP" "$SUBMIT"

    # Read rather than trusted: notarytool can finish waiting on a rejection, and the log is
    # the only place that says which binary Apple objected to.
    OUT=$(xcrun notarytool submit "$SUBMIT" --keychain-profile "$PROFILE" --wait 2>&1) || true
    echo "$OUT"

    if ! echo "$OUT" | grep -q "status: Accepted"; then
        ID=$(echo "$OUT" | awk '/^  id:/ {print $2; exit}')
        [ -n "$ID" ] && xcrun notarytool log "$ID" --keychain-profile "$PROFILE"
        echo "Notarization failed; nothing to share." >&2
        exit 1
    fi

    # Stapled, so a Mac that is offline the first time it opens the app still finds the
    # ticket in the bundle instead of asking Apple.
    xcrun stapler staple "$APP"
    rm -f "$SUBMIT"
fi

if [ -n "$IDENTITY" ]; then
    rm -f "$SHARE"
    ditto -c -k --keepParent "$APP" "$SHARE"
fi

# A copy, not a symlink: a link would point Finder into a OneDrive-synced folder and let
# every rebuild mutate the installed app while it runs.
if [ -n "$INSTALL" ]; then
    rm -rf /Applications/LocalScribe.app
    cp -R "$APP" /Applications/
    echo
    echo "Installed /Applications/LocalScribe.app"
fi

echo
echo "Built $APP"

if [ -n "$PROFILE" ]; then
    echo "To share: $SHARE (signed, notarized, stapled)"
elif [ -n "$IDENTITY" ]; then
    echo "Signed but not notarized: $SHARE is blocked on another Mac until its user allows it in System Settings > Privacy & Security."
fi
echo "First launch downloads the models (about 2.8 GiB) into"
echo "~/Library/Application Support/LocalScribe/models — not into the bundle, whose"
echo "signature must survive."
