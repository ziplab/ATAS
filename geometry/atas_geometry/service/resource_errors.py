"""Explicit infrastructure failures; never classify geometry rejection as retryable."""
from pathlib import Path


def native_resource_code(message):
    """Recognize infrastructure initialization failures, not numeric rejection.

    cuSOLVER creation failure is not proof of OOM; retain its distinct code.
    SVD non-convergence, NaNs and invalid geometry stay non-retryable here.
    """
    text = str(message).lower()
    if "cuda error: an illegal instruction was encountered" in text:
        return "cuda_native_runtime_failed"
    if any(token in text for token in ("cuda out of memory", "torch.outofmemoryerror", "cuda error: out of memory")):
        return "cuda_out_of_memory"
    if "cusolver_status_internal_error" in text and "cusolverdncreate" in text:
        return "cuda_linalg_initialization_failed"
    return None


class ResourceUnavailableError(RuntimeError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.error_code = code
        self.resource_details = details

    def fields(self):
        return {"failure_category": "resource", "error_code": self.error_code,
                "retryable": True, "resource_details": self.resource_details}


def log_resource_error(path, *, physical_gpu_index, stage, start_offset=0):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(max(start_offset, path.stat().st_size - 32768, 0))
        tail = handle.read().decode("utf-8", errors="replace")
    lines = [line for line in tail.splitlines() if native_resource_code(line)]
    if not lines:
        return None
    code = native_resource_code(lines[-1])
    description = {"cuda_out_of_memory": "CUDA memory exhausted",
                   "cuda_native_runtime_failed": "CUDA native runtime crashed"}.get(code, "CUDA linear-algebra runtime initialization failed")
    return ResourceUnavailableError(code,
        f"{description} on physical GPU {physical_gpu_index}; reacquire a suitable GPU before retry",
        physical_gpu_index=physical_gpu_index, logical_gpu_index=0, stage=stage,
        log_path=str(path), evidence=lines[-1][:1600])


def error_fields(exc, *, physical_gpu_index=None, stage=None):
    if isinstance(exc, ResourceUnavailableError):
        return exc.fields()
    code = native_resource_code(exc)
    if code:
        return ResourceUnavailableError(code, str(exc),
            physical_gpu_index=physical_gpu_index, stage=stage, evidence=str(exc)[:1600]).fields()
    return {}
