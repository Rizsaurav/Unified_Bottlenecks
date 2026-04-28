import re
import pandas as pd
import os

def parse_powermetrics(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None

    current_entry = {}
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Let's peek at the first few lines to see why they don't match
        lines = f.readlines()
        if not lines:
            print("The file is empty!")
            return None
            
        print(f"File loaded. Total lines: {len(lines)}")

        for line in lines:
            clean_line = line.strip()
            
            # Use a more relaxed search for the start of a block
            if "Sampled" in clean_line and "activity" in clean_line:
                if "Timestamp" in current_entry and "Power_mW" in current_entry:
                    data.append(current_entry)
                
                current_entry = {}
                # Grab the date/time inside the parentheses
                time_match = re.search(r"\((.*?)\)", clean_line)
                current_entry["Timestamp"] = time_match.group(1) if time_match else "Unknown"

            elif "Combined Power" in clean_line:
                p_match = re.findall(r"(\d+)", clean_line)
                if p_match:
                    current_entry["Power_mW"] = int(p_match[0])

            elif "GPU HW active residency" in clean_line:
                g_match = re.search(r"(\d+\.?\d*)", clean_line)
                if g_match:
                    current_entry["GPU_Residency_Pct"] = float(g_match.group(1))

        # Catch the last one
        if "Timestamp" in current_entry and "Power_mW" in current_entry:
            data.append(current_entry)

    return pd.DataFrame(data)

file_path = "results/logs/m2_video_trace_20260419_001300.txt"
df = parse_powermetrics(file_path)

if df is not None and not df.empty:
    df.to_csv('vid1_m2_performance_results.csv', index=False)
    print(f"--- SUCCESS ---")
    print(f"Extracted {len(df)} samples.")
    print(f"Peak Power: {df['Power_mW'].max()} mW")
else:
    print("--- DEBUG INFO ---")
    with open(file_path, 'r') as f:
        head = [next(f) for _ in range(5)]
        print("First 5 lines of file look like this:")
        for i, l in enumerate(head):
            print(f"{i}: {repr(l)}")