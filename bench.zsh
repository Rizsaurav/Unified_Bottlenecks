#!/bin/zsh

# 1. Setup Folders
mkdir -p results/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/logs/m2_trace_$TIMESTAMP.txt"

echo "Starting Automated Benchmark: $TIMESTAMP"

# 2. Start Hardware Logging in the background
# We use & to run it in the background and save the Process ID (PID)
sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal -i 500 > "$LOG_FILE" &
POWER_PID=$!

# Give the logger a second to stabilize
sleep 2

# 3. Run the AI Vision Model
# Change the image path if needed
./llama.cpp/build/bin/llama-minicpmv-cli \
  -m llama.cpp/models/InternVL2_5-4B-Q4_K_M.gguf \
  --mmproj llama.cpp/models/mmproj-InternVL2_5-4B-f16.gguf \
  --image ./images/test_v1.jpg \
  -p "Describe the image in extreme detail." \
  -t 8 --color

# 4. Cleanup: Kill the power logger
sudo kill $POWER_PID

echo "Benchmark Complete. Data saved to: $LOG_FILE"
