#!/bin/zsh
mkdir -p ../data/raw_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="../data/raw_logs/m2_text_baseline_8bit_$TIMESTAMP.txt"

sudo -v
sudo powermetrics --samplers cpu_power,gpu_power,thermal -i 500 > "$LOG_FILE" &
POWER_PID=$!

sleep 2

# 4. Sustained Text-only inference
./llama.cpp/build/bin/llama-mtmd-cli \
  -m llama.cpp/models/InternVL2_5-4B-Q8_0.gguf \
  --mmproj llama.cpp/models/mmproj-InternVL2_5-4B-f16.gguf \
  -p "Generate a list of 500 distinct reasons why Unified Memory is the future of computing, then write a detailed 10-page technical manual on M2 architectural bottlenecks." \
  -n 2048 \
  -t 8 --temp 0.7