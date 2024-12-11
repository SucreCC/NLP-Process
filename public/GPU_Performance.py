import pynvml
import time

# 初始化 NVML
pynvml.nvmlInit()

# 设置日志文件路径
log_file = "experiments/5-gpu_monitor_log.txt"

# 获取显卡数量
device_count = pynvml.nvmlDeviceGetCount()

# 开始监控
try:
    with open(log_file, "w") as f:  # 打开日志文件
        f.write("torch.cuda.set_per_process_memory_fraction(0.80)")
        f.write("torch.backends.cudnn.benchmark = True")
        f.write("CUDA_LAUNCH_BLOCKING = 0")
        f.write("time.sleep(0.5)")
        f.write("batch size = 32")

        f.write("GPU Monitoring Log\n")
        f.write("=" * 40 + "\n")

        while True:  # 实时监控
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 获取每块显卡的句柄
                name = pynvml.nvmlDeviceGetName(handle)  # 获取显卡名称
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 显存信息
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)  # 利用率信息

                log_entry = (
                        # f"GPU {i}: {name.decode('utf-8')}\n"
                        f"  Memory Used: {memory_info.used / 1024 ** 2:.2f} MB\n"
                        f"  Memory Total: {memory_info.total / 1024 ** 2:.2f} MB\n"
                        f"  GPU Utilization: {utilization.gpu}%\n"
                        f"  Memory Utilization: {utilization.memory}%\n"
                        f"-" * 40 + "\n"
                )

                # 写入文件并打印
                f.write(log_entry)
                print(log_entry)

            # 每隔5秒记录一次
            time.sleep(10)
except KeyboardInterrupt:
    print("Monitoring stopped by user.")
finally:
    # 释放 NVML
    pynvml.nvmlShutdown()
    print(f"Logs saved to {log_file}")
