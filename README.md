# Devoxx Belgium 2026: demos for "Tap into the NVIDIA ecosystem from Java"

Every demo of the talk that runs on this machine, one command each. (Apache Flink + cuDF runs on the other machine.)

```bash
ssh <this machine>
cd demoDevoxx
bash check.sh            # before the talk: runs everything once, prints OK/FAIL per demo (~40 s)

bash demoHybrid.sh       # slides 14-16  Java kernel -> cuBLAS -> Java kernel in one task graph; CUDA graph replay   (~3 s)
bash demoTile.sh         # slides 17-19  threads + CUDA Tile + cuBLAS in one CUDA graph; the tile GEMM ladder      (~6 s)
bash demoJitllm.sh       # live demo 1   jitLLM generates on the GPU, then jitLLM vs llama.cpp                      (~20 s)
```

Each script prints the command before running it and waits for **Enter** between steps (`NO_PAUSE=1` to skip).

## The demos

| script | variants | what to point at |
|---|---|---|
| `demoHybrid.sh` | | Per task: the Java kernels (`scale`, `bias`) and `cublasSgemv` all on CUDA, every iteration `correct`. The first iteration's cuBLAS time is its one-time initialisation. Then `withCUDAGraph()`: about 9× faster steady state (≈310 → 35 µs). |
| `demoTile.sh` | | The capture: `@Parallel` scale, a `TileContext` GEMM, `cublasSgemv` and a Java `biasRelu` between `BEGIN_CAPTURE` and `END_CAPTURE`, then replays. Then the ladder: tile shape alone takes the tile GEMM from 2.1 to 1.2 ms, and one launch hint passes the hand-optimised kernel. Wall clock includes dispatch; slide 19's TFLOP/s are kernel time from nsys. |
| `demoJitllm.sh` | `chat`, `bench` | Generation in pure Java on the GPU (~190 tok/s, Llama-3.2-1B F16). Then Qwen3-0.6B F16: prefill with cuBLAS (`--with-native-libraries`), decode with Java kernels replayed as a CUDA graph, and llama.cpp on the same GPU and GGUF. |

Other models: `bash models.sh` lists every GGUF on the machine, about 30, among them Qwen3-1.7B F16
(`~/jcon/models/Qwen3-1.7B-f16.gguf`, the slide's second prefill model), Qwen3-4B/8B F16, Llama-3.2-1B/3B,
Llama-3-8B, gemma-4-E2B, Phi-3, Mistral-7B, DeepSeek-R1-Distill-Qwen-1.5B/7B and Qwen3.8-27B Q4_0. Use one with
`MODEL_CHAT=<path> bash demoJitllm.sh chat` or `MODEL_BENCH=<path> bash demoJitllm.sh bench`.

## Measured on this machine (RTX 4090, 2026-10-01)

| | result |
|---|---|
| Hybrid: CUDA graph vs plain execute | 34.7 vs 312 µs, 8.98× |
| Tile ladder, n = 2048 (wall clock) | KernelContext optimised 1176 µs · TileContext 128×128×64 + hint 1123 µs · cuBLAS 1060 µs |
| jitLLM Qwen3-0.6B F16, prefill pp512 b512 | jitLLM 64,818–65,340 t/s · llama.cpp 60,923–64,545 t/s (± up to 9k between its runs) |
| jitLLM Qwen3-0.6B F16, decode tg128 | jitLLM 399 t/s (CUDA graphs) · llama.cpp 515 t/s |

jitLLM is the latest `main` (`d875f083`, 2026-09-30). llama.cpp is the local build `f172be756`.

## What is behind each script

| demo | uses |
|---|---|
| `demoHybrid.sh`, `demoTile.sh` | `~/tornadovm-devoxx2026-cuda-demos` (demos 04, 07, 20, 25) on the released TornadoVM 7.0.0 (SDKMAN, JDK 25) |
| `demoJitllm.sh` | `./jitllm` (beehive-lab/jitllm `main`), built against its TornadoVM develop build (JDK 21); `~/llama.cpp-ref/build/bin/llama-bench` |

All paths are in `env.sh`. `bash setup.sh` (network) updates jitLLM to the latest `main` and rebuilds it.

## If something fails on stage

* **`check.sh` says FAIL:** open the log it names; most failures are a missing environment (`env.sh` paths).
* **jitLLM is slow on the first run:** the first run JIT-compiles the kernels. Run `bash demoJitllm.sh chat` once
  before the talk.
* **GPU busy:** check `nvidia-smi` for stray Java processes from an interrupted demo.
