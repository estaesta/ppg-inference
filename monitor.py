import psutil
import time

def get_max_cpu_usage(process_name, duration):
    """
    Monitors the CPU usage of a process with the given name and returns the maximum CPU usage observed within the duration.
    
    :param process_name: The name of the process to monitor.
    :param duration: Duration in seconds to monitor the process.
    :return: Maximum CPU usage observed as a percentage.
    """
    max_cpu_usage = 0.0
    start_time = time.time()
    pids = []

    while time.time() - start_time < duration:
        # Iterate through all processes and find the ones that match the process name
        for proc in psutil.process_iter(['pid', 'name']):
            if process_name in proc.info['name']:
                pids.append(proc.info['pid'])
                print(f"Found process '{process_name}' with PID: {proc.info['pid']}")

        for pid in pids:
            try:
                proc = psutil.Process(pid)
                cpu_usage = proc.cpu_percent(interval=1)
                max_cpu_usage = max(max_cpu_usage, cpu_usage)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    return max_cpu_usage

# Example usage
process_name = 'gunicorn'
duration = 10  # Monitor for 60 seconds

max_cpu = get_max_cpu_usage(process_name, duration)
print(f"The maximum CPU usage of '{process_name}' during the last {duration} seconds was: {max_cpu}%")
