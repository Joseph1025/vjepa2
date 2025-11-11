#!/bin/bash

# Script to convert AV1 videos to H.264 for decord compatibility
# This will create new files with _h264.mp4 suffix

SOURCE_DIR="/home/zexi/Robo_data/lerobot_0801_tac/videos/chunk-000"

echo "Converting tactile1 videos..."
for f in "$SOURCE_DIR/observation.tactile1"/*.mp4; do
    if [[ ! "$f" =~ _h264\.mp4$ ]]; then
        outfile="${f%.mp4}_h264.mp4"
        if [ ! -f "$outfile" ]; then
            echo "Converting $(basename "$f")..."
            ffmpeg -i "$f" -c:v libx264 -preset fast -crf 23 "$outfile" -y -loglevel error
        else
            echo "Skipping $(basename "$f") (already converted)"
        fi
    fi
done

echo ""
echo "Converting tactile2 videos..."
for f in "$SOURCE_DIR/observation.tactile2"/*.mp4; do
    if [[ ! "$f" =~ _h264\.mp4$ ]]; then
        outfile="${f%.mp4}_h264.mp4"
        if [ ! -f "$outfile" ]; then
            echo "Converting $(basename "$f")..."
            ffmpeg -i "$f" -c:v libx264 -preset fast -crf 23 "$outfile" -y -loglevel error
        else
            echo "Skipping $(basename "$f") (already converted)"
        fi
    fi
done

echo ""
echo "✅ Conversion complete!"
echo ""
echo "Now update your CSV files to use _h264.mp4 files:"
echo "  sed -i 's/\\.mp4/_h264.mp4/g' tactile1_0801.csv"
echo "  sed -i 's/\\.mp4/_h264.mp4/g' tactile2_0801.csv"

