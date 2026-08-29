#!/usr/bin/env bash
#
# Put the AstraVigil launcher on the desktop.
#
#   ./deploy/install_kiosk.sh
#
# Writes a .desktop entry with absolute paths (the desktop will not run a
# relative Exec), marks it executable, and tells the file manager it is
# trusted - without which Raspberry Pi OS shows "Untrusted application
# launcher" instead of running it.
#
# Uninstall:  rm ~/Desktop/AstraVigil.desktop ~/.local/share/applications/AstraVigil.desktop

set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
LAUNCHER="$REPO/deploy/astravigil-kiosk.sh"

# Raspberry Pi OS localises the desktop folder; ask rather than assume.
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
[ -n "$DESKTOP_DIR" ] && [ -d "$DESKTOP_DIR" ] || DESKTOP_DIR="$HOME/Desktop"
mkdir -p "$DESKTOP_DIR" "$HOME/.local/share/applications"

chmod +x "$LAUNCHER" "$REPO/deploy/stop-kiosk.sh"

ARGS="${ASTRAVIGIL_ARGS:-}"
write_entry() {
    cat >"$1" <<ENTRY
[Desktop Entry]
Type=Application
Version=1.0
Name=AstraVigil
GenericName=Counter-UAS console
Comment=Start the sensor and open the operator console fullscreen
Exec=env ASTRAVIGIL_ARGS="$ARGS" "$LAUNCHER"
Path=$REPO
Icon=camera-video
Terminal=false
StartupNotify=true
Categories=Utility;Security;
Keywords=drone;thermal;perimeter;
ENTRY
    chmod +x "$1"
}

write_entry "$DESKTOP_DIR/AstraVigil.desktop"
write_entry "$HOME/.local/share/applications/AstraVigil.desktop"

# PCManFM (Pi OS) refuses to run desktop launchers it does not trust.
gio set "$DESKTOP_DIR/AstraVigil.desktop" metadata::trusted true 2>/dev/null || true
# Older file managers used a different key.
gio set "$DESKTOP_DIR/AstraVigil.desktop" metadata::xfce-exe-checksum \
    "$(sha256sum "$DESKTOP_DIR/AstraVigil.desktop" | cut -d' ' -f1)" 2>/dev/null || true

command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true

echo "Installed:"
echo "  $DESKTOP_DIR/AstraVigil.desktop"
echo "  $HOME/.local/share/applications/AstraVigil.desktop"
echo
echo "Pipeline arguments: ${ARGS:-<defaults from astravigil-kiosk.sh>}"
echo "  To change them, re-run with e.g.:"
echo "    ASTRAVIGIL_ARGS='--source synthetic --scenario intrusion' $0"
echo
echo "Before trusting this on stage, TEST THE WAY OUT:"
echo "  1. double-click the icon"
echo "  2. press Ctrl+Shift+Esc three times - a counter should appear"
echo "  3. if it does not, run:  $REPO/deploy/stop-kiosk.sh"
echo
if ! command -v chromium-browser >/dev/null 2>&1 \
   && ! command -v chromium >/dev/null 2>&1; then
    echo "WARNING: no Chromium found. Install it first:"
    echo "    sudo apt install -y chromium-browser"
fi
if ! command -v unclutter >/dev/null 2>&1; then
    echo "Optional, hides the mouse pointer:  sudo apt install -y unclutter"
fi
