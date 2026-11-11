"""A misbehaving worker: consumes stdin frames but never responds.

Used by tests to simulate a hung worker (e.g. a frozen CUDA context) and
exercise the proxy's deadline + single-retry + degraded ladder.
"""
import sys

for _raw in sys.stdin.buffer:
    pass  # swallow every frame; never answer
