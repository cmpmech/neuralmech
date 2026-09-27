#!/bin/bash
set -euo pipefail

# archives and the URL manifest stay in external_data/abc; paths are resolved
# relative to this script so it can be run from anywhere
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ABC_DIR="$SCRIPT_DIR/../../external_data/abc"
cd "$ABC_DIR"

cat stl2_v00.txt | xargs -n 2 -P 8 sh -c 'wget --no-check-certificate $0 -O stl2/$1'
