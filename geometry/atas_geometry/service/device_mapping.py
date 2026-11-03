"""Keep nvidia-smi physical GPU indices distinct from CUDA-visible device indices."""
import json
import os


def gpu_selector(physical_index):
    return json.loads(os.getenv("GEOMETRY_GPU_UUIDS", "{}" )).get(str(physical_index), str(physical_index))


def device_index(physical_index):
    selector = gpu_selector(physical_index)
    if not selector.startswith("GPU-"):
        return int(physical_index)  # Deployments without an explicit UUID mapping.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if selector not in visible:
        raise RuntimeError(f"physical GPU {physical_index} not explicitly visible")
    return visible.index(selector)
