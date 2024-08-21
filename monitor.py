import psutil
import time

def get_max_cpu_and_memory_usage(process_names, duration):
    """
    Monitors the total CPU usage and memory usage of processes with the given names,
    and returns the maximum CPU usage, average memory usage, and maximum cumulative memory usage observed within the duration.
    
    :param process_names: List of process names to monitor.
    :param duration: Duration in seconds to monitor the processes.
    :return: A tuple containing (maximum CPU usage, average memory usage, maximum cumulative memory usage).
    """
    max_cpu_usage = 0.0
    max_cumulative_memory_usage = 0.0
    total_memory_usage = 0.0
    count_memory_readings = 0
    start_time = time.time()

    while True:
        try:
            if time.time() - start_time >= duration:
                break

            # Collect the total memory usage for processes with the given names
            total_cpu_usage = 0.0
            current_total_memory_usage = 0.0

            for proc in psutil.process_iter(['pid', 'name']):
                if any(name in proc.info['name'] for name in process_names):
                    try:
                        cpu_usage = proc.cpu_percent(interval=1)
                        total_cpu_usage += cpu_usage

                        memory_info = proc.memory_info()
                        memory_usage = memory_info.rss  # Resident Set Size (RSS) in bytes
                        current_total_memory_usage += memory_usage
                        max_cumulative_memory_usage = max(max_cumulative_memory_usage, current_total_memory_usage)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        # Handle cases where the process may no longer exist or cannot be accessed
                        continue

            # Update the maximum CPU usage observed
            max_cpu_usage = max(max_cpu_usage, total_cpu_usage)
            count_memory_readings += 1
            total_memory_usage += current_total_memory_usage

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break

    average_memory_usage = total_memory_usage / count_memory_readings if count_memory_readings > 0 else 0

    # Convert memory usage from bytes to megabytes for readability
    max_cumulative_memory_usage_mb = max_cumulative_memory_usage / (1024 * 1024)
    average_memory_usage_mb = average_memory_usage / (1024 * 1024)

    return max_cpu_usage, average_memory_usage_mb, max_cumulative_memory_usage_mb

# Example usage
process_names = ['gunicorn', 'flask']
duration = 10 * 60  # Monitor for 10 minutes

max_cpu, avg_memory, max_memory = get_max_cpu_and_memory_usage(process_names, duration)
print(f"The maximum total CPU usage of processes {process_names} during the last {duration} seconds was: {max_cpu}%")
print(f"The average memory usage of processes {process_names} during the last {duration} seconds was: {avg_memory:.2f} MB")
print(f"The maximum cumulative memory usage of processes {process_names} during the last {duration} seconds was: {max_memory:.2f} MB")
