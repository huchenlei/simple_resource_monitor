import logging
import signal
import json
import time
import pynvml

logger = logging.getLogger(__name__)


class GPUInfo:
    """
    This class is responsible for getting information from GPU (ONLY).
    """

    cuda = False
    pynvmlLoaded = False
    cudaDevicesFound = 0
    switchGPU = True
    switchVRAM = True
    switchTemperature = True
    gpus = []
    gpusUtilization = []
    gpusVRAM = []
    gpusTemperature = []

    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.pynvmlLoaded = True
        except pynvml.NVMLError as e:
            self.pynvmlLoaded = False
            logger.error("Could not init pynvml. %s", e)

        if self.pynvmlLoaded and pynvml.nvmlDeviceGetCount() > 0:
            self.cudaDevicesFound = pynvml.nvmlDeviceGetCount()

            logger.info("GPU/s:")

            for deviceIndex in range(self.cudaDevicesFound):
                deviceHandle = pynvml.nvmlDeviceGetHandleByIndex(deviceIndex)
                gpuName = "Unknown GPU"

                try:
                    gpuName = pynvml.nvmlDeviceGetName(deviceHandle)
                    try:
                        gpuName = gpuName.decode("utf-8", errors="ignore")
                    except AttributeError:
                        pass

                except UnicodeDecodeError as e:
                    gpuName = "Unknown GPU (decoding error)"
                    print(f"UnicodeDecodeError: {e}")

                logger.info("%s) %s", deviceIndex, gpuName)

                self.gpus.append(
                    {
                        "index": deviceIndex,
                        "name": gpuName,
                    }
                )

                # same index as gpus, with default values
                self.gpusUtilization.append(True)
                self.gpusVRAM.append(True)
                self.gpusTemperature.append(True)

            self.cuda = True
            logger.info("NVIDIA Driver: %s", pynvml.nvmlSystemGetDriverVersion())
        else:
            logger.warning("No GPU with CUDA detected.")

    def getStatus(self):
        gpuUtilization = -1
        gpuTemperature = -1
        vramUsed = -1
        vramTotal = -1
        vramPercent = -1

        gpuType = ""
        gpus = []

        if self.pynvmlLoaded and self.cuda:
            for deviceIndex in range(self.cudaDevicesFound):
                deviceHandle = pynvml.nvmlDeviceGetHandleByIndex(deviceIndex)

                gpuUtilization = 0
                vramPercent = 0
                vramUsed = 0
                vramTotal = 0
                gpuTemperature = 0

                # GPU Utilization
                # https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html#structnvmlUtilization__t
                if self.switchGPU and self.gpusUtilization[deviceIndex]:
                    try:
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(deviceHandle)
                        gpuUtilization = utilization.gpu
                    except Exception as e:
                        if str(e) == "Unknown Error":
                            logger.error(
                                "For some reason, pynvml is not working in a laptop with only battery, try to connect and turn on the monitor"
                            )
                        else:
                            logger.error("Could not get GPU utilization. %s", e)

                        logger.error("Monitor of GPU is turning off (not on UI!)")
                        self.switchGPU = False

                # VRAM
                if self.switchVRAM and self.gpusVRAM[deviceIndex]:
                    # Torch or pynvml?, pynvml is more accurate with the system, torch is more accurate with comfyUI
                    memory = pynvml.nvmlDeviceGetMemoryInfo(deviceHandle)
                    vramUsed = memory.used
                    vramTotal = memory.total
                    vramPercent = vramUsed / vramTotal * 100

                # Temperature
                if self.switchTemperature and self.gpusTemperature[deviceIndex]:
                    try:
                        gpuTemperature = pynvml.nvmlDeviceGetTemperature(
                            deviceHandle, 0
                        )
                    except Exception as e:
                        logger.error(
                            "Could not get GPU temperature. Turning off this feature. %s",
                            str(e),
                        )
                        self.switchTemperature = False

                gpus.append(
                    {
                        "gpu_utilization": gpuUtilization,
                        "gpu_temperature": gpuTemperature,
                        "vram_total": vramTotal,
                        "vram_used": vramUsed,
                        "vram_used_percent": vramPercent,
                    }
                )

        return {
            "device_type": gpuType,
            "gpus": gpus,
        }


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


if __name__ == "__main__":
    killer = GracefulKiller()
    gpu_info = GPUInfo()
    results = []

    while not killer.kill_now:
        status = gpu_info.getStatus()
        results.append(status)
        print(status)
        time.sleep(1)

    with open("status.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
