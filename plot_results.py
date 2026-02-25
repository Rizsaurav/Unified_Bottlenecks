import matplotlib.pyplot as plt
import re
import glob

def parse_power(file_path):
    timestamps, cpu, gpu = [], [], []
    with open(file_path, 'r') as f:
        data = f.read()
        # Find all power readings using Regex
        cpu_vals = re.findall(r"CPU Power: (\d+) mW", data)
        gpu_vals = re.findall(r"GPU Power: (\d+) mW", data)

        # Convert to Watts and plot
        cpu = [int(x)/1000 for x in cpu_vals]
        gpu = [int(x)/1000 for x in gpu_vals]

    plt.figure(figsize=(10, 5))
    plt.plot(cpu, label='CPU Power (W)', color='blue')
    plt.plot(gpu, label='GPU Power (W)', color='green')
    plt.title("M2 Unified Bottleneck: Vision vs Text Phase")
    plt.xlabel("Time (Samples @ 500ms)")
    plt.ylabel("Power (Watts)")
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path.replace('.txt', '.png'))
    print(f"Graph saved as {file_path.replace('.txt', '.png')}")

# Run it on your latest log
latest_log = max(glob.glob("results/logs/*.txt"))
parse_power(latest_log)
