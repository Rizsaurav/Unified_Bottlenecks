#!/bin/zsh

# 1. Setup Folders
mkdir -p ../data/raw_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="../data/raw_logs/m2_trace_8bit_$TIMESTAMP.txt"

echo "Starting Automated Benchmark: $TIMESTAMP"

# 2. Cache sudo credentials
# This prevents the background logger from hanging while waiting for a password
sudo -v

# 3. Start Hardware Logging in the background
# We don't use -n here so we can capture the full duration of the AI run
sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal -i 500 > "$LOG_FILE" &
POWER_PID=$!

# Give the logger a moment to initialize the file
sleep 2

# 4. Run the AI Vision Model
# I added -t 8 to ensure you are stressing all M2 cores for your paper
./llama.cpp/build/bin/llama-mtmd-cli \
  -m llama.cpp/models/InternVL2_5-4B-Q8_0.gguf \
  --mmproj llama.cpp/models/mmproj-InternVL2_5-4B-f16.gguf \
  --image ./images/control/test_v1.jpg \
  -p "Describe the image in extreme detail." \
  -t 8

# 5. Cleanup & Flush Data
echo "\nFinishing... Flushing hardware logs to disk."

# kill -2 (SIGINT) is the "polite" way to stop powermetrics and force it to save its buffer
sudo kill -2 $POWER_PID

# The 'sync' command tells the macOS filesystem to write all pending data to the SSD now
sync

echo "Benchmark Complete."
echo "Log File: $LOG_FILE"
echo "Log Size: $(du -h "$LOG_FILE" | cut -f1)"