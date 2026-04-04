"""vLLM service discovery, launch, and health monitoring utilities.

Extracted from test scripts to allow reuse across the project.
All functions accept a ``log_fn`` callback (default ``print``) so callers
can inject their own logger (e.g. elapsed-time prefixed output).
"""
import os
import select
import subprocess
import sys
import time
from typing import Callable, Optional

import requests
from openai import OpenAI

# ── Project-level defaults ───────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def _load_dotenv(root: str = _PROJECT_ROOT) -> None:
    """Load .env file into os.environ (setdefault, no overwrite)."""
    env_path = os.path.join(root, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
    "/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
DEFAULT_QWEN35_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B"
)
DEFAULT_CACHE_ROOT = os.path.join(_PROJECT_ROOT, "cache")
DEFAULT_SERVING_INFO_DIR = os.path.join(DEFAULT_CACHE_ROOT, ".serving")
DEFAULT_SERVING_IP_FILE = os.path.join(DEFAULT_SERVING_INFO_DIR, "serving_ip.txt")
DEFAULT_SERVING_LOG_FILE = os.path.join(DEFAULT_SERVING_INFO_DIR, "vllm.log")
DEFAULT_RL_SH_PATH = os.path.join(_PROJECT_ROOT, "scripts", "rl.sh")
DEFAULT_VLLM_PORT = 8000


# Patterns in rlaunch stderr that indicate the worker will never start.
_FATAL_PATTERNS = [
    "insufficient group quota",
    "does not pass quotaCheck",
    "denied the request",
    "Insufficient resources",
    "tasks failed",
]


def _default_log(msg: str = ""):
    print(msg, flush=True)


# ── Health & Discovery ────────────────────────────────────────────────────────

def probe_vllm(base_url: str, timeout: float = 5.0,
               log_fn: Callable = _default_log) -> bool:
    """Check if a vLLM service is alive at *base_url* by querying ``/v1/models``."""
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
    """Probe a list of candidate base_urls, return the first alive one."""
    for url in candidates:
        log_fn(f"  Probing {url} ...")
        if probe_vllm(url, log_fn=log_fn):
            return url
    return None


def read_serving_ip(serving_ip_file: str = None) -> Optional[str]:
    """Read the serving IP written by a previously launched GPU job."""
    serving_ip_file = serving_ip_file or DEFAULT_SERVING_IP_FILE
    if os.path.isfile(serving_ip_file):
        ip = open(serving_ip_file).read().strip()
        if ip:
            return ip
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
    """Factory for OpenAI SDK client pointing at a vLLM endpoint."""
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)


# ── Subprocess Monitoring ─────────────────────────────────────────────────────

def _drain_stderr(proc: subprocess.Popen) -> str:
    """Non-blocking read of all currently available stderr from *proc*."""
    chunks = []
    while True:
        ready, _, _ = select.select([proc.stderr], [], [], 0)
        if not ready:
            break
        chunk = proc.stderr.read1(4096) if hasattr(proc.stderr, "read1") else proc.stderr.read(4096)
        if not chunk:
            break
        chunks.append(chunk.decode("utf-8", errors="replace"))
    return "".join(chunks)


def check_rlaunch_health(proc: subprocess.Popen,
                         log_fn: Callable = _default_log) -> None:
    """Read rlaunch stderr and abort early on fatal scheduling errors."""
    stderr_text = _drain_stderr(proc)
    if stderr_text:
        for line in stderr_text.strip().splitlines():
            log_fn(f"  [rlaunch] {line.strip()}")
        for pattern in _FATAL_PATTERNS:
            if pattern in stderr_text:
                log_fn(f"  FATAL: Worker scheduling failed — found '{pattern}' in rlaunch output.")
                proc.terminate()
                sys.exit(1)
    ret = proc.poll()
    if ret is not None and ret != 0:
        remaining = proc.stderr.read().decode("utf-8", errors="replace")
        if remaining:
            for line in remaining.strip().splitlines():
                log_fn(f"  [rlaunch] {line.strip()}")
        log_fn(f"  FATAL: rlaunch exited with code {ret}.")
        sys.exit(1)


# ── Launch & Wait ─────────────────────────────────────────────────────────────

def _resolve_qwen35_model() -> Optional[str]:
    """Find the latest Qwen3.5-9B snapshot in the HuggingFace cache."""
    import glob
    snapshots = sorted(glob.glob(os.path.join(DEFAULT_QWEN35_MODEL_PATH, "snapshots", "*")))
    return snapshots[-1] if snapshots else None


def launch_serving(
    model_path: str = None,
    rl_sh_path: str = None,
    serving_info_dir: str = None,
    serving_ip_file: str = None,
    serving_log_file: str = None,
    vllm_port: int = DEFAULT_VLLM_PORT,
    gpu: int = 2,
    tp: Optional[int] = None,
    allowed_local_media_path: str = "/",
    qwen35: bool = False,
    log_fn: Callable = _default_log,
) -> subprocess.Popen:
    """Launch a vLLM serving job on a GPU node via ``rl.sh``.

    The GPU job writes its IP to *serving_ip_file* on a shared filesystem
    so the caller can discover it.  Returns the Popen process for monitoring.

    When *qwen35* is True (or auto-detected from model_path):
      - Adds ``--reasoning-parser qwen3`` for thinking mode support
      - Sets CUDA_HOME from .env for flashinfer JIT compilation
      - Prepends nvcc to PATH
    """
    if model_path is None:
        if qwen35:
            model_path = _resolve_qwen35_model() or "Qwen/Qwen3.5-9B"
        else:
            model_path = DEFAULT_MODEL_PATH

    # Auto-detect Qwen3.5 from model path
    name_lower = model_path.lower().replace("-", "").replace("_", "")
    is_qwen35 = qwen35 or "qwen3.5" in name_lower or "qwen35" in name_lower

    rl_sh_path = rl_sh_path or DEFAULT_RL_SH_PATH
    serving_info_dir = serving_info_dir or DEFAULT_SERVING_INFO_DIR
    serving_ip_file = serving_ip_file or DEFAULT_SERVING_IP_FILE
    serving_log_file = serving_log_file or DEFAULT_SERVING_LOG_FILE
    os.makedirs(serving_info_dir, exist_ok=True)
    if os.path.isfile(serving_ip_file):
        os.remove(serving_ip_file)
    if tp is None:
        tp = gpu

    # Build env prefix for CUDA_HOME (needed for flashinfer JIT on Qwen3.5)
    env_prefix = ""
    if is_qwen35:
        cuda_home = os.environ.get("CUDA_HOME", "")
        if cuda_home:
            env_prefix = (
                f'export CUDA_HOME={cuda_home}; '
                f'export PATH=$CUDA_HOME/bin:$PATH; '
            )

    # Build vLLM flags
    vllm_flags = (
        f'--host 0.0.0.0 --port {vllm_port} '
        f'--tensor-parallel-size {tp} '
        f'--max-model-len {"65536" if is_qwen35 else "32768"} '
        f'--allowed-local-media-path {allowed_local_media_path}'
    )
    if is_qwen35:
        vllm_flags += ' --reasoning-parser qwen3'

    inner_script = (
        f'{env_prefix}'
        f'IP=$(hostname -I | awk \'{{print $1}}\'); '
        f'echo "$IP" > {serving_ip_file}; '
        f'echo "[serving] Node IP: $IP, starting vLLM ..." | tee {serving_log_file}; '
        f'exec vllm serve {model_path} '
        f'{vllm_flags} '
        f'2>&1 | tee -a {serving_log_file}'
    )
    cmd = [rl_sh_path, "-gpu", str(gpu), "--", "bash", "-c", inner_script]
    log_fn(f"Launching vLLM serving with {gpu} GPUs (TP={tp}, qwen3.5={is_qwen35}) ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log_fn(f"  Launched rlaunch process (pid={proc.pid})")
    return proc


def wait_for_serving(
    proc: subprocess.Popen,
    serving_ip_file: str = None,
    vllm_port: int = DEFAULT_VLLM_PORT,
    timeout: int = 600,
    poll_interval: int = 10,
    log_fn: Callable = _default_log,
) -> str:
    """Wait for the GPU job to write its IP and for vLLM to become ready.

    Two phases:
    1. Wait for *serving_ip_file* to appear (GPU node coordination).
    2. Poll ``/v1/models`` until vLLM responds.

    Returns the ``base_url`` (e.g. ``http://1.2.3.4:8000/v1``).
    """
    serving_ip_file = serving_ip_file or DEFAULT_SERVING_IP_FILE
    log_fn(f"Waiting for serving to start (timeout={timeout}s) ...")
    start = time.time()

    # Phase 1: wait for IP file from GPU node
    ip = None
    while time.time() - start < timeout:
        check_rlaunch_health(proc, log_fn=log_fn)
        ip = read_serving_ip(serving_ip_file)
        if ip:
            log_fn(f"  GPU node IP: {ip}")
            break
        time.sleep(poll_interval)
    else:
        check_rlaunch_health(proc, log_fn=log_fn)
        log_fn("ERROR: Timed out waiting for GPU node to write its IP.")
        proc.terminate()
        sys.exit(1)

    # Phase 2: wait for vLLM /v1/models to respond
    base_url = f"http://{ip}:{vllm_port}/v1"
    while time.time() - start < timeout:
        check_rlaunch_health(proc, log_fn=log_fn)
        if probe_vllm(base_url, timeout=10, log_fn=log_fn):
            log_fn(f"  vLLM is ready at {base_url}")
            return base_url
        log_fn(f"  vLLM not ready yet, retrying in {poll_interval}s ...")
        time.sleep(poll_interval)

    check_rlaunch_health(proc, log_fn=log_fn)
    log_fn("ERROR: Timed out waiting for vLLM to become ready.")
    proc.terminate()
    sys.exit(1)
