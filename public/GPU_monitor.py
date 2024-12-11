import pynvml
import time
import os
import glob
import threading

class GPUMonitor:
    def __init__(self, log_dir):
        """
        Initialize the GPUMonitor class with a specific log directory.

        Args:
            log_dir (str): Path to the directory where logs will be stored.
        """
        self.log_dir = log_dir
        self._clear_logs()
        self.running = False
        self.monitor_thread = None

    def _clear_logs(self):
        """Clear all files in the log directory."""
        if os.path.exists(self.log_dir):
            files = glob.glob(os.path.join(self.log_dir, "*"))
            for f in files:
                try:
                    os.remove(f)
                except PermissionError:
                    print(f"Warning: Unable to delete file {f} because it is in use.")
        else:
            os.makedirs(self.log_dir, exist_ok=True)

    @staticmethod
    def initialize_nvml():
        """Initialize NVML library."""
        pynvml.nvmlInit()

    @staticmethod
    def shutdown_nvml():
        """Shutdown NVML library."""
        pynvml.nvmlShutdown()

    @staticmethod
    def get_device_count():
        """Get the number of GPU devices available."""
        return pynvml.nvmlDeviceGetCount()

    def _monitor(self, task_name, interval, max_iterations):
        """
        Internal method to perform GPU monitoring in a separate thread.

        Args:
            task_name (str): Name of the monitoring task.
            interval (int): Time interval between each logging in seconds.
            max_iterations (int): Maximum number of times to collect GPU information (None for infinite).
        """
        GPUMonitor.initialize_nvml()
        device_count = GPUMonitor.get_device_count()
        iterations = 0
        buffer = []  # Buffer to store log entries

        log_file = os.path.join(self.log_dir, f"{task_name}_log.txt")

        try:
            with open(log_file, "w") as f:
                f.write("\n\n")  # Add two blank lines
                f.write("=" * 40 + "\n")  # Separator line
                f.write(f"Log Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"GPU Monitoring Task: {task_name}\n")
                f.write("GPU Monitoring Log\n")
                f.write("=" * 40 + "\n")

                while self.running:
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                        gpu_utilization = utilization.gpu
                        memory_utilization = (memory_info.used / memory_info.total) * 100

                        # Generate log entry
                        log_entry = (
                                f"Task: {task_name}\n"
                                f"GPU {i}: {name}\n"
                                f"  Memory Used: {memory_info.used / 1024 ** 2:.2f} MB\n"
                                f"  Memory Total: {memory_info.total / 1024 ** 2:.2f} MB\n"
                                f"  GPU Utilization: {gpu_utilization}%\n"
                                f"  Memory Utilization: {memory_utilization:.2f}%\n"
                                f"-" * 40 + "\n"
                        )

                        buffer.append(log_entry)  # Add log entry to buffer

                    if len(buffer) >= max_iterations:  # Write to file if buffer size matches max_iterations
                        f.writelines(buffer)
                        buffer = []  # Clear buffer
                        f.flush()

                    iterations += 1

                    if max_iterations and iterations >= max_iterations:
                        break

                    time.sleep(interval)

                if buffer:  # Write remaining buffer content to file
                    f.writelines(buffer)
                    f.flush()
        except KeyboardInterrupt:
            print("Monitoring stopped by user.")
        finally:
            GPUMonitor.shutdown_nvml()

    def start_monitoring(self, task_name="Unknown Task", interval=10, max_iterations=20):
        """
        Start GPU monitoring in a separate thread.

        Args:
            task_name (str): Name of the monitoring task.
            interval (int): Time interval between each logging in seconds.
            max_iterations (int): Maximum number of times to collect GPU information (None for infinite).
        """
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor, args=(task_name, interval, max_iterations), daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop the GPU monitoring task."""
        if not self.running:
            return

        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
