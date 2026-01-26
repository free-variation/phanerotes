#!/bin/bash
# Extract audio features at video frame rate using ffmpeg
# Usage: audio_features.sh <audio_file> <fps> <output.csv>

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <audio_file> <fps> [output.tsv]"
    exit 1
fi

AUDIO="$1"
FPS="$2"
BASENAME=$(basename "$AUDIO" | sed 's/\.[^.]*$//')
OUTPUT="${3:-${BASENAME}_features.tsv}"

# Get sample rate
SAMPLE_RATE=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 "$AUDIO")

# Calculate hop size for desired fps
HOP=$(echo "$SAMPLE_RATE / $FPS" | bc)
OVERLAP=$(echo "scale=4; 1 - ($HOP / 2048)" | bc)

TMPDIR=$(mktemp -d)
SPECTRAL="$TMPDIR/spectral.txt"
AMPLITUDE="$TMPDIR/amplitude.txt"

echo "Sample rate: $SAMPLE_RATE, Hop: $HOP, Overlap: $OVERLAP" >&2

# Extract spectral stats
ffmpeg -i "$AUDIO" -af "aspectralstats=win_size=2048:overlap=$OVERLAP:measure=centroid+spread+flux+flatness+rolloff,ametadata=print:file=$SPECTRAL" -f null - 2>/dev/null

# Extract amplitude stats (force frame size to match fps, reset every frame)
ffmpeg -i "$AUDIO" -af "asetnsamples=n=$HOP,astats=metadata=1:reset=1,ametadata=print:file=$AMPLITUDE" -f null - 2>/dev/null

# Parse and combine into CSV
awk '
BEGIN {
    OFS="\t"
    print "frame\ttime\tcentroid_l\tcentroid_r\tspread_l\tspread_r\tflux_l\tflux_r\tflatness_l\tflatness_r\trolloff_l\trolloff_r\trms_l\trms_r\tpeak_l\tpeak_r"
}
FNR==NR {
    # First file: spectral stats
    if (/^frame:/) {
        split($0, a, /[: ]+/)
        frame = a[2]
        time = a[6]
    }
    if (/\.1\.centroid=/) { split($0, a, "="); centroid_l[frame] = a[2] }
    if (/\.2\.centroid=/) { split($0, a, "="); centroid_r[frame] = a[2] }
    if (/\.1\.spread=/) { split($0, a, "="); spread_l[frame] = a[2] }
    if (/\.2\.spread=/) { split($0, a, "="); spread_r[frame] = a[2] }
    if (/\.1\.flux=/) { split($0, a, "="); flux_l[frame] = a[2] }
    if (/\.2\.flux=/) { split($0, a, "="); flux_r[frame] = a[2] }
    if (/\.1\.flatness=/) { split($0, a, "="); flatness_l[frame] = a[2] }
    if (/\.2\.flatness=/) { split($0, a, "="); flatness_r[frame] = a[2] }
    if (/\.1\.rolloff=/) { split($0, a, "="); rolloff_l[frame] = a[2] }
    if (/\.2\.rolloff=/) { split($0, a, "="); rolloff_r[frame] = a[2] }
    times[frame] = time
    max_frame = frame
    next
}
{
    # Second file: amplitude stats
    if (/^frame:/) {
        split($0, a, /[: ]+/)
        frame = a[2]
    }
    if (/Overall\.RMS_level=/) { split($0, a, "="); rms[frame] = a[2] }
    if (/1\.RMS_level=/) { split($0, a, "="); rms_l[frame] = a[2] }
    if (/2\.RMS_level=/) { split($0, a, "="); rms_r[frame] = a[2] }
    if (/1\.Peak_level=/) { split($0, a, "="); peak_l[frame] = a[2] }
    if (/2\.Peak_level=/) { split($0, a, "="); peak_r[frame] = a[2] }
}
END {
    for (f = 0; f <= max_frame; f++) {
        # Only output frames that have both spectral and amplitude data
        if (centroid_l[f] != "" && rms_l[f] != "") {
            printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
                f, times[f], \
                centroid_l[f], centroid_r[f], \
                spread_l[f], spread_r[f], \
                flux_l[f], flux_r[f], \
                flatness_l[f], flatness_r[f], \
                rolloff_l[f], rolloff_r[f], \
                rms_l[f], rms_r[f], \
                peak_l[f], peak_r[f]
        }
    }
}
' "$SPECTRAL" "$AMPLITUDE" > "$OUTPUT"

rm -rf "$TMPDIR"

echo "Wrote $(tail -n +2 "$OUTPUT" | wc -l | tr -d ' ') frames to $OUTPUT" >&2
