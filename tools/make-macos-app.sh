#!/bin/sh
# Builds LocalScribe.app for Apple silicon.
#
# Self-contained, carrying the Windows decision across: the app is a folder someone is handed
# and must not care what .NET the machine has. Signed ad-hoc — enough for personal use and for
# the microphone permission to stick to the bundle; distribution needs a real identity and
# notarization, decided when someone other than the builder is going to run it.
set -e

cd "$(dirname "$0")/.."

PUBLISH=build/publish-mac
APP=build/LocalScribe.app

dotnet publish src/LocalScribe.Desktop -c Release -r osx-arm64 --self-contained -o "$PUBLISH"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp -R "$PUBLISH/". "$APP/Contents/MacOS/"
cp tools/macos/Info.plist "$APP/Contents/"

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

codesign --force --deep --sign - "$APP"

echo
echo "Built $APP"
echo "First launch downloads the models (about 2.8 GiB) into"
echo "~/Library/Application Support/LocalScribe/models — not into the bundle, whose"
echo "signature must survive."
