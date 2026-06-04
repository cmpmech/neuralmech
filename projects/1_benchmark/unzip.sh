#!/bin/bash
set -euo pipefail

# .7z archives stay in external_data/abc/stl2; the extracted/renamed .stl files
# become repo data under data/abc/geometry/stl. Paths are relative to this script.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE_DIR="$SCRIPT_DIR/../../external_data/abc/stl2"
OUTPUT_DIR="$SCRIPT_DIR/../../data/abc/geometry/stl"
mkdir -p "$OUTPUT_DIR"

# Optional first argument: max number of .7z archives to process (default: all)
MAX_ARCHIVES="${1:-0}"

# Extract each archive flat into the output directory
n=0
for archive in "$ARCHIVE_DIR"/abc_*_stl2_v00.7z; do
    if [ "$MAX_ARCHIVES" -gt 0 ] && [ "$n" -ge "$MAX_ARCHIVES" ]; then
        break
    fi
    echo "Extracting: $archive"
    7z e "$archive" -o"$OUTPUT_DIR" -y >/dev/null
    n=$((n + 1))
done

# 7z recreates the archive's directory entries as empty folders -- drop them
find "$OUTPUT_DIR" -mindepth 1 -type d -empty -delete

# Rename sequentially in sorted order: 000000001_abc.stl, 000000002_abc.stl, ...
i=1
find "$OUTPUT_DIR" -maxdepth 1 -type f -name '*.stl' | sort | while read -r f; do
    printf -v newname "%09d_abc.stl" "$i"
    mv "$f" "$OUTPUT_DIR/$newname"
    i=$((i + 1))
done

echo "Done. Files extracted and renamed in: $OUTPUT_DIR"
