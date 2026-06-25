"""vLLM service discovery, local launch, and health monitoring utilities."""
import os
import subprocess
import time
from typing import Callable, Optional

import requests
from openai import OpenAI

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_MODEL_PATH = os.getenv("VIDIFY_VLLM_MODEL", "Qwen/Qwen3.5-9B")
DEFAULT_CACHE_ROOT = os.path.join(_PROJECT_ROOT, "cache")
DEFAULT_VLLM_PORT = int(os.getenv("VIDIFY_VLLM_PORT", "8000"))


def _default_log(msg: str = ""):
    print(msg, flush=True)


def probe_vllm(base_url: str, timeout: float = 5.0,
               log_fn: Callable = _default_log) -> bool:
    """Check if a vLLM service is alive at *base_url* via ``/v1/models``."""
    try:
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        resp = requests.get(f"{url}/models", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            log_fn(f"  Found serving at {url} with models: {models}")
            return True
    except Exception:
        pass
    return False


def find_existing_service(candidates: list,
                          log_fn: Callable = _default_log) -> Optional[str]:
    """Probe candidate base URLs and return the first responding endpoint."""
    for url in candidates:
        log_fn(f"  Probing {url} ...")
        if probe_vllm(url, log_fn=log_fn):
            normalized = url.rstrip("/")
            return normalized if normalized.endswith("/v1") else f"{normalized}/v1"
    return None


def get_model_name(base_url: str) -> str:
    """Query ``/v1/models`` and return the first model ID."""
    url = base_url.rstrip("/")
    resp = requests.get(f"{url}/models", timeout=10)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    if not models:
        raise RuntimeError("No models available on the serving endpoint")
    return models[0]["id"]


def make_client(base_url: str, timeout: float = 120.0) -> OpenAI:
    """Create an OpenAI SDK client pointed at a vLLM endpoint."""
    return OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                  timeout=timeout)


def launch_serving(
    model_path: str = None,
    vllm_port: int = DEFAULT_VLLM_PORT,
    gpu: int = 1,
    tp: Optional[int] = None,
    allowed_local_media_path: str = None,
    qwen35: bool = False,
    log_fn: Callable = _default_log,
) -> subprocess.Popen:
    """Launch a local ``vllm serve`` process.

    This helper is intentionally local-only. For managed GPU/NPU environments,
    start vLLM with your platform's scheduler and pass ``--api-base`` to the
    scripts that need a serving endpoint.
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    tp = tp or gpu or 1
    allowed_local_media_path = allowed_local_media_path or os.path.join(_PROJECT_ROOT, "cache")

    name_lower = model_path.lower().replace("-", "").replace("_", "")
    is_qwen35 = qwen35 or "qwen3.5" in name_lower or "qwen35" in name_lower

    cmd = [
        "vllm", "serve", model_path,
        "--host", "0.0.0.0",
        "--port", str(vllm_port),
        "--tensor-parallel-size", str(tp),
        "--max-model-len", "65536" if is_qwen35 else "32768",
        "--allowed-local-media-path", allowed_local_media_path,
    ]
    if is_qwen35:
        cmd += ["--reasoning-parser", "qwen3"]

    log_fn(f"Launching local vLLM serving on port {vllm_port} (TP={tp}) ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_fn(f"  Launched vLLM process (pid={proc.pid})")
    return proc


def wait_for_serving(
    proc: subprocess.Popen,
    vllm_port: int = DEFAULT_VLLM_PORT,
    timeout: int = 600,
    poll_interval: int = 10,
    log_fn: Callable = _default_log,
) -> str:
    """Wait for a local vLLM process to respond and return its base URL."""
    base_url = f"http://localhost:{vllm_port}/v1"
    start = time.time()
    log_fn(f"Waiting for serving at {base_url} (timeout={timeout}s) ...")

    while time.time() - start < timeout:
        ret = proc.poll()
        if ret is not None and ret != 0:
            raise RuntimeError(f"vLLM exited with code {ret}")
        if probe_vllm(base_url, timeout=5, log_fn=log_fn):
            return base_url
        time.sleep(poll_interval)

    raise TimeoutError(f"vLLM did not become ready within {timeout}s")
