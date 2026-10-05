#!/usr/bin/env bash
# The Hybrid API, live dashboard (~20 s): one TaskGraph mixing NVIDIA cuFFT library tasks with Java kernels that
# TornadoVM JIT-compiles, on shared device buffers. A low-pass filter sweeps over a noisy signal, one execution of
# the whole graph per frame: the pipeline with each task's GPU time (TornadoVM's profiler), the input, the spectrum,
# the output, the bytes that cross PCIe, every frame checked; then the same graph replayed as a CUDA graph.
#   bash fancyHybridLive.sh        [enter] at the end to leave (NO_PAUSE=1 to exit by itself); terminal >= 124 x 40
set -euo pipefail
source "$(dirname "$0")/env.sh"
use_tornadovm_7
export TORNADOVM_HOME JAVA_HOME
exec python3 "$DEMO_ROOT/hybrid/live.py" "$(mktemp -d)"
