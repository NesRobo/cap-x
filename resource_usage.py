#!/usr/bin/env python3
"""Quick check of CPU, memory, and NVIDIA GPU availability on Linux."""
import csv
import shutil
import subprocess
import time


def cpu_percent():
    def read():
        values = list(map(int, open("/proc/stat").readline().split()[1:]))
        return sum(values), values[3] + values[4]

    total1, idle1 = read()
    time.sleep(1)
    total2, idle2 = read()
    return 100 * (1 - (idle2 - idle1) / (total2 - total1))


print(f"CPU: {cpu_percent():.1f}% used ({__import__('os').cpu_count()} logical cores)")

memory = subprocess.run(["free", "-h"], capture_output=True, text=True)
if memory.returncode == 0:
    print("\nMemory:")
    print("\n".join(memory.stdout.splitlines()[:2]))

if not shutil.which("nvidia-smi"):
    print("\nNVIDIA GPU: nvidia-smi not found")
else:
    gpu = subprocess.run([
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv",
    ], capture_output=True, text=True)
    print("\nGPU:")
    print(gpu.stdout.strip() if gpu.returncode == 0 else "Could not read GPU status")

    processes = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv",
    ], capture_output=True, text=True)
    print("\nGPU processes (user, PID, process, GPU memory):")
    rows = list(csv.reader(processes.stdout.splitlines()))[1:]
    if not rows:
        print("None reported")
    else:
        for pid, name, memory_used in rows:
            owner = subprocess.run(
                ["ps", "-o", "user=", "-p", pid], capture_output=True, text=True
            ).stdout.strip() or "unknown"
            print(f"{owner}, PID {pid}, {name}, {memory_used}")

print("\nCPU processes using the most CPU:")
processes = subprocess.run(
    ["ps", "-eo", "user,pid,pcpu,comm", "--sort=-pcpu"],
    capture_output=True, text=True,
)
print("\n".join(processes.stdout.splitlines()[:7]))
