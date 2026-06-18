#!/usr/bin/env bash
# Manual verify for the single-card overlay + history navigation.
#
# Prereqs: a Wayland session, OPENAI_API_KEY set, kitetsu.toml with
# [teleprompter] enabled, and teleprompter/teleprompter.toml present.
#
# Run the daemon in one terminal:
#   cargo run -p kitetsu --features teleprompter -- daemon
# Then run this script in another to drive it and read the checklist.
set -euo pipefail

ctl="$(dirname "$(readlink -f "$0")")/../../kitetsu-ctl"
step() { printf '\n== %s ==\n' "$1"; }
pause() { read -rp "  press Enter when checked… " _; }

step "ONE card only"
echo "  EXPECT: exactly one card on screen (not two stacked)."
pause

step "process → suggestion #1"
"$ctl" process
echo "  EXPECT: header 'Fetching response…' then a suggestion appears."
pause

step "process → suggestion #2"
"$ctl" process
echo "  EXPECT: a second, newer suggestion replaces the first."
pause

step "backward → older"
"$ctl" backward
echo "  EXPECT: card shows suggestion #1 again."
pause

step "forward → newer"
"$ctl" forward
echo "  EXPECT: card shows suggestion #2 again (newest)."
pause

step "backward past oldest / forward past newest are no-ops"
"$ctl" backward; "$ctl" backward; "$ctl" forward; "$ctl" forward
echo "  EXPECT: stops at #1 going back, at #2 going forward — no blank card."
pause

step "drag + resize persist"
echo "  Drag the card and resize from the corner, then restart the daemon."
echo "  EXPECT: it reopens at the moved/resized geometry (layout.toml)."
pause

echo "done."
