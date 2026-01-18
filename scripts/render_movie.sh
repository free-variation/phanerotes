#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <generator> [jobs]"
    echo "  generator - Executable that outputs phanerotes commands (one per line)"
    echo "  jobs      - Number of parallel jobs (default: 8, use 1 for sequential)"
    exit 1
fi

GENERATOR="$1"
JOBS="${2:-8}"

# Derive name from generator (e.g., movie-scripts/generate_chromatic.py -> chromatic)
NAME=$(basename "$GENERATOR" | sed 's/^generate_//' | sed 's/\.[^.]*$//')

PHAN_SCRIPT="movie-scripts/${NAME}.phan"
FRAMES_DIR="frames/${NAME}"
MOVIE_FILE="movies/${NAME}.mp4"

echo "==> Generating script from $GENERATOR"
"$GENERATOR" > "$PHAN_SCRIPT"

echo "==> Clearing old frames in $FRAMES_DIR"
rm -f "$FRAMES_DIR"/*.bmp

if [ "$JOBS" -eq 1 ]; then
    echo "==> Rendering frames sequentially"
    ./phanerotes < "$PHAN_SCRIPT"
else
    echo "==> Rendering frames with $JOBS parallel jobs"
    parallel --pipe -N1 -j"$JOBS" ./phanerotes < "$PHAN_SCRIPT"
fi

echo "==> Encoding movie to $MOVIE_FILE"
ffmpeg -y -framerate 30 -i "$FRAMES_DIR/frame_%03d.bmp" -c:v libx264 -pix_fmt yuv420p "$MOVIE_FILE"

echo "==> Done: $MOVIE_FILE"
