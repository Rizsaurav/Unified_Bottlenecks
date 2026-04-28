#!/bin/zsh

# 1. Setup Folders
mkdir -p ../data/raw_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="../data/raw_logs/m2_video_trace_$TIMESTAMP.txt"

echo "Starting Automated VIDEO Benchmark: $TIMESTAMP"

# 2. Cache sudo credentials
sudo -v

# 3. Start Hardware Logging
# Using -i 500 (half-second resolution) to catch thermal throttling transitions
sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal -i 500 > "$LOG_FILE" &
POWER_PID=$!

# Give the logger a moment to initialize
sleep 2

# 4. Run the AI Video Model
# Note: Using --video instead of --image. 
# This processes 5 frames to create a sustained hardware load
for i in {1..5}; do
  echo "Processing Frame $i..."
  ./llama.cpp/build/bin/llama-mtmd-cli \
    -m llama.cpp/models/InternVL2_5-4B-Q4_K_M.gguf \
    --mmproj llama.cpp/models/mmproj-InternVL2_5-4B-f16.gguf \
    --image ./images/control/test_v1.jpg \
    -p "Describe the movement in this frame." \
    -t 8 --temp 0
done

# 5. Cleanup & Flush Data
echo "\nVideo processing finished. Flushing hardware logs..."

# Standard interrupt to ensure powermetrics closes the file pointer properly
sudo kill -2 $POWER_PID

# Force disk write
sync

echo "Benchmark Complete."
echo "Video Log File: $LOG_FILE"
echo "Final Log Size: $(du -h "$LOG_FILE" | cut -f1)"