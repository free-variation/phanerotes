#!/bin/bash
set -e

MODE="${1:-gui}"  # gui or text
INPUT="${2:-projects/fort-greene-2/make_video.phan}"
THREADS="${3:-8}"
WAIT="${4:-45}"
DURATION="${5:-60}"

if [ "$MODE" != "gui" ] && [ "$MODE" != "text" ]; then
    echo "Usage: $0 [gui|text] [input] [threads] [wait] [duration]"
    echo "  gui  - use Instruments (opens GUI)"
    echo "  text - use sample (outputs profile.txt)"
    exit 1
fi

echo "Building..."
fpm build --profile release --flag "-g -march=native -ffast-math -fopenmp"

# Find and sign the binary that fpm will run
BINARY=$(find build -path "*/app/phanerotes" -type f -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)

if [ -z "$BINARY" ]; then
    echo "Error: Could not find binary"
    exit 1
fi

echo "Signing $BINARY for profiling..."
cat > /tmp/entitlements.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.get-task-allow</key>
    <true/>
</dict>
</plist>
EOF

codesign -s - --entitlements /tmp/entitlements.plist -f "$BINARY"
rm /tmp/entitlements.plist

echo "Starting program via fpm run..."
OMP_NUM_THREADS=$THREADS fpm run --profile release --flag "-g -march=native -ffast-math -fopenmp" < "$INPUT" &
PID=$!

echo "PID: $PID (fpm wrapper)"

# Wait for fpm to spawn the actual binary
for i in {1..10}; do
    sleep 1
    REAL_PID=$(pgrep -n phanerotes 2>/dev/null || echo "")
    if [ -n "$REAL_PID" ]; then
        echo "Actual binary PID: $REAL_PID"
        PID=$REAL_PID
        break
    fi
done

echo "Waiting ${WAIT}s for setup..."
sleep $WAIT

if ! kill -0 $PID 2>/dev/null; then
    echo "Process already finished"
    exit 1
fi

if [ "$MODE" = "text" ]; then
    echo "Sampling for ${DURATION}s..."
    sample $PID $DURATION -f profile.txt
    kill $PID 2>/dev/null
    echo ""
    echo "Results in profile.txt"
    echo ""
    echo "=== Top of call graph ==="
    grep -A 200 "Call graph:" profile.txt | head -100
else
    echo "Recording for ${DURATION}s..."
    rm -rf profile.trace
    xcrun xctrace record --template "Time Profiler" --attach $PID --time-limit ${DURATION}s --output profile.trace
    echo "Opening trace..."
    open profile.trace
fi
