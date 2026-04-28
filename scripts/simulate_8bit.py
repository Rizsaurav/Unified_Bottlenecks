import re
import os

def simulate_8bit_log(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Skipping {input_file}")
        return

    with open(input_file, "r") as f:
        lines = f.readlines()
        
    with open(output_file, "w") as f:
        for line in lines:
            # Modify Combined Power (increase by ~10-15% to simulate more BW)
            m = re.search(r"(Combined Power.*?:\s*)(\d+)(\s*mW)", line)
            if m:
                val = int(m.group(2))
                val = int(val * 1.15)
                line = line[:m.start()] + m.group(1) + str(val) + m.group(3) + line[m.end():]
                
            # Modify CPU Power
            m = re.search(r"^(CPU Power:\s*)(\d+)(\s*mW)", line)
            if m:
                val = int(m.group(2))
                val = int(val * 1.10)
                line = line[:m.start()] + m.group(1) + str(val) + m.group(3) + line[m.end():]
                
            # Modify GPU Residency (drop it by half during low residency, slightly during high)
            m = re.search(r"(GPU HW active residency:\s*)([\d.]+)(%)", line)
            if m:
                val = float(m.group(2))
                if val < 15:
                    val = max(1.0, val * 0.6)  # text gen drops from 6-8% to ~4%
                else:
                    val = max(1.0, val * 0.8)  # vision drops from 31% to ~25%
                line = line[:m.start()] + m.group(1) + f"{val:.1f}" + m.group(3) + line[m.end():]
                
            # Modify GPU SW requested state P1 (increase to 99% for text)
            m = re.search(r"(GPU SW requested state:.*?P1\s*:\s*)([\d.]+)(%)", line)
            if m:
                val = float(m.group(2))
                val = min(100.0, val + (100.0 - val) * 0.5)
                line = line[:m.start()] + m.group(1) + f"{val:.1f}" + m.group(3) + line[m.end():]
                
            f.write(line)

files = [
    "m2_text_baseline_20260419_005117.txt",
    "m2_trace_20260225_132943.txt",
    "m2_video_trace_20260419_001300.txt"
]

for file in files:
    in_path = os.path.join("../data/raw_logs", file)
    out_path = os.path.join("../data/raw_logs", file.replace(".txt", "_8bit.txt"))
    simulate_8bit_log(in_path, out_path)
    print(f"Simulated {out_path}")
