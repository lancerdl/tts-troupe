"""
Lightweight CUDA readiness check for detecting broken or frozen CUDA contexts (e.g. after PC sleep).
"""

import concurrent.futures
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # GPU-free environments (e.g. CI or the phase-1 test suite)
    torch = None

logger = logging.getLogger(__name__)


def _ping_cuda(device: str = "cuda") -> bool:
    """Minimal GPU allocation and sync to verify CUDA context responsiveness."""
    x = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    del x
    return True


def is_cuda_healthy(device: str = "cuda", timeout_seconds: float = 2.0) -> bool:
    """
    Check if CUDA is functional and responsive.

    Tier 1 (Non-risky pre-checks):
      - Verifies CUDA availability and device presence.
      - If CUDA has never been initialized in this process, the state is pristine; returns True immediately.

    Tier 2 (Timed ping probe):
      - If CUDA was already initialized, executes a fast allocation and synchronization
        inside a worker thread bounded by `timeout_seconds`.
      - Catches driver errors or thread hangs caused by power-state transitions (e.g. Windows sleep).

    Returns:
        bool: True if CUDA is healthy/ready, False if unavailable, broken, or frozen.
    """
    # --- Tier 1: Non-risky pre-checks (zero GPU execution) ---
    if torch is None:
        logger.warning("PyTorch is not installed; CUDA is unavailable.")
        return False

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available on this system.")
        return False

    if torch.cuda.device_count() <= 0:
        logger.warning("CUDA is available but no CUDA devices were found.")
        return False

    # If CUDA has never been initialized in this process, no context exists yet to go stale.
    if not torch.cuda.is_initialized():
        logger.debug("CUDA has not been initialized yet; context is pristine.")
        return True

    # --- Tier 2: Active responsiveness check for initialized context ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_ping_cuda, device)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.error(
                "CUDA ping timed out after %.1fs (possible sleep/driver freeze).",
                timeout_seconds,
            )
            return False
        except Exception as e:
            logger.error("CUDA ping failed with error: %s", e)
            return False
