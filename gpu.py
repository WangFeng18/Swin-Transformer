import pynvml
import time
pynvml.nvmlInit()

def printNvidiaGPU(gpu_id):
    # get GPU temperature
    gpu_device = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    temperature = pynvml.nvmlDeviceGetTemperature(gpu_device, pynvml.NVML_TEMPERATURE_GPU)
    # get GPU memory total
    totalMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).total
    # get GPU memory used
    usedMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used

    performance = pynvml.nvmlDeviceGetPerformanceState(gpu_device)

    powerUsage = pynvml.nvmlDeviceGetPowerUsage(gpu_device)
    powerState = pynvml.nvmlDeviceGetPowerState(gpu_device)
    FanSpeed = pynvml.nvmlDeviceGetFanSpeed(gpu_device)
    PersistenceMode = pynvml.nvmlDeviceGetPersistenceMode(gpu_device)
    UtilizationRates = pynvml.nvmlDeviceGetUtilizationRates(gpu_device)

    print("MemoryInfo：{0}M/{1}M，使用率：{2}%".format("%.1f" % (usedMemory / 1024 / 1024), "%.1f" % (totalMemory / 1024 / 1024), "%.1f" % (usedMemory/totalMemory*100)))
    print("Temperature：{0}摄氏度".format(temperature))
    print("Performance：{0}".format(performance))
    print("PowerState: {0}".format(powerState))
    print("PowerUsage: {0}".format(powerUsage / 1000))
    print("FanSpeed: {0}".format(FanSpeed))
    print("PersistenceMode: {0}".format(PersistenceMode))
    print("UtilizationRates: {0}".format(UtilizationRates.gpu))
    time.sleep(1)
    
while (1):
    printNvidiaGPU(0) # 此处以0号gpu为例