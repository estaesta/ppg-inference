import psutil
import time

def get_max_cpu_and_memory_usage(process_names, duration):
    """
    Monitors the total CPU usage and memory usage of processes with the given names,
    and returns the maximum CPU usage, average memory usage, and maximum memory usage observed within the duration.
    
    :param process_names: List of process names to monitor.
    :param duration: Duration in seconds to monitor the processes.
    :return: A tuple containing (maximum CPU usage, average memory usage, maximum memory usage).
    """
    max_cpu_usage = 0.0
    max_memory_usage = 0.0
    total_memory_usage = 0.0
    count_memory_readings = 0
    start_time = time.time()

    # Dictionary to store process IDs and their respective memory usage over time
    pids = set()

    while True:
        try:
            if time.time() - start_time >= duration:
                break

            # Collect PIDs of processes that match any of the given names
            current_pids = set()
            for proc in psutil.process_iter(['pid', 'name']):
                if any(name in proc.info['name'] for name in process_names):
                    current_pids.add(proc.info['pid'])
                    if proc.info['pid'] not in pids:
                        print(f"Found process with PID: {proc.info['pid']}")

            # Update the list of monitored PIDs
            pids = current_pids

            # Calculate total CPU usage and memory usage of all matching processes
            total_cpu_usage = 0.0
            total_memory_usage = 0.0
            for pid in pids:
                try:
                    proc = psutil.Process(pid)
                    # Calculate CPU usage
                    cpu_usage = proc.cpu_percent(interval=1)
                    total_cpu_usage += cpu_usage
                    # Calculate memory usage
                    memory_info = proc.memory_info()
                    memory_usage = memory_info.rss  # Resident Set Size (RSS) in bytes
                    total_memory_usage += memory_usage
                    max_memory_usage = max(max_memory_usage, memory_usage)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # Handle cases where the process may no longer exist or cannot be accessed
                    pids.discard(pid)

            # Update the maximum CPU usage observed
            max_cpu_usage = max(max_cpu_usage, total_cpu_usage)
            count_memory_readings += 1
            total_memory_usage += total_memory_usage

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break

    average_memory_usage = total_memory_usage / count_memory_readings if count_memory_readings > 0 else 0

    # Convert memory usage from bytes to megabytes for readability
    max_memory_usage_mb = max_memory_usage / (1024 * 1024)
    average_memory_usage_mb = average_memory_usage / (1024 * 1024)

    return max_cpu_usage, average_memory_usage_mb, max_memory_usage_mb

# Example usage
process_names = ['gunicorn', 'flask']
duration = 10 * 60  # Monitor for x minutes

max_cpu, avg_memory, max_memory = get_max_cpu_and_memory_usage(process_names, duration)
print(f"The maximum total CPU usage of processes {process_names} during the last {duration} seconds was: {max_cpu}%")
print(f"The average memory usage of processes {process_names} during the last {duration} seconds was: {avg_memory:.2f} MB")
print(f"The maximum memory usage of processes {process_names} during the last {duration} seconds was: {max_memory:.2f} MB")
