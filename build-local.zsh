#!/bin/zsh
# Builds the plugin with nothing but the Xcode Command Line Tools, against the OBS that is installed on this Mac.
# The CMake build (see README) remains the way to produce release packages; it needs a full Xcode install.
#
#   ./build-local.zsh            build into build_local/
#   ./build-local.zsh install    build, then replace the plugin in OBS's per-user plugin folder (old one is backed up)

set -euo pipefail

ROOT=${0:A:h}
OBS_APP=${OBS_APP:-/Applications/OBS.app}
SIMDE_TAG=v0.8.2

NAME=$(plutil -extract name raw "$ROOT/buildspec.json")
VERSION=$(plutil -extract version raw "$ROOT/buildspec.json")
BUNDLE_ID=$(plutil -extract platformConfig.macos.bundleId raw "$ROOT/buildspec.json")
OBS_VERSION=$(plutil -extract CFBundleShortVersionString raw "$OBS_APP/Contents/Info.plist")

DEPS=$ROOT/.deps
OUT=$ROOT/build_local
BUNDLE=$OUT/$NAME.plugin

# libobs headers matching the installed OBS, and the SIMD shim they include.
if [[ ! -d $DEPS/obs-studio-$OBS_VERSION/libobs ]]; then
    git clone -q --depth 1 --branch "$OBS_VERSION" --filter=blob:none --sparse \
        https://github.com/obsproject/obs-studio.git "$DEPS/obs-studio-$OBS_VERSION"
    git -C "$DEPS/obs-studio-$OBS_VERSION" sparse-checkout set libobs
fi
if [[ ! -d $DEPS/simde ]]; then
    git clone -q --depth 1 --branch "$SIMDE_TAG" https://github.com/simd-everywhere/simde.git "$DEPS/simde"
fi

rm -rf "$OUT"
mkdir -p "$OUT/gen" "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"

print "#pragma once\n#define OBS_RELEASE_CANDIDATE 0\n#define OBS_BETA 0" > "$OUT/gen/obsconfig.h"
sed -e "s/@CMAKE_PROJECT_NAME@/$NAME/" -e "s/@CMAKE_PROJECT_VERSION@/$VERSION/" \
    "$ROOT/src/plugin-support.c.in" > "$OUT/gen/plugin-support.c"

clang -bundle -O2 -g -std=gnu23 -fno-objc-arc -mmacosx-version-min=13.0 \
    -Wall -Wextra -Wno-unused-parameter \
    -I "$ROOT/src" -I "$OUT/gen" -isystem "$DEPS/obs-studio-$OBS_VERSION/libobs" -isystem "$DEPS/simde" \
    -F "$OBS_APP/Contents/Frameworks" -framework libobs \
    -framework Foundation -framework CoreVideo -framework CoreMedia -framework Vision \
    -o "$BUNDLE/Contents/MacOS/$NAME" \
    "$ROOT/src/plugin-main.m" "$OUT/gen/plugin-support.c"

cp -R "$ROOT/data/" "$BUNDLE/Contents/Resources/"

cat > "$BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key><string>en</string>
	<key>CFBundleDisplayName</key><string>$NAME</string>
	<key>CFBundleExecutable</key><string>$NAME</string>
	<key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
	<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
	<key>CFBundleName</key><string>$NAME</string>
	<key>CFBundlePackageType</key><string>BNDL</string>
	<key>CFBundleShortVersionString</key><string>$VERSION</string>
	<key>CFBundleVersion</key><string>$(date +%s)</string>
	<key>LSMinimumSystemVersion</key><string>13.0</string>
</dict>
</plist>
EOF

codesign --force --sign - "$BUNDLE"
print "Built $BUNDLE ($(lipo -archs "$BUNDLE/Contents/MacOS/$NAME"), against OBS $OBS_VERSION)"

if [[ ${1:-} == install ]]; then
    PLUGINS="$HOME/Library/Application Support/obs-studio/plugins"
    BACKUPS="$HOME/Library/Application Support/obs-studio/plugins-backup"
    mkdir -p "$PLUGINS" "$BACKUPS"
    # Move rather than overwrite: a running OBS keeps using the old files until it is restarted.
    STAMP=$(date +%Y%m%d-%H%M%S)
    for item in "$PLUGINS/$NAME.plugin" "$PLUGINS/$NAME.plugin.dSYM"; do
        [[ -e $item ]] && mv "$item" "$BACKUPS/${item:t}.$STAMP"
    done
    cp -R "$BUNDLE" "$PLUGINS/"
    print "Installed to $PLUGINS/$NAME.plugin (previous version in $BACKUPS). Restart OBS to load it."
fi
