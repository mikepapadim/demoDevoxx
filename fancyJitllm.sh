#!/usr/bin/env bash
# jitLLM, the fancy version of demoJitllm.sh (live demo 1, slides 20-22): live generation in pure Java on the GPU, then
# jitLLM vs llama.cpp as bar charts. About 50 seconds.
#   bash fancyJitllm.sh [chat|bench]     NO_PAUSE=1 to skip the [enter] between acts
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$DEMO_ROOT/fancy.sh"
use_jitllm
cd "$JITLLM_DIR"
mode="${1:-all}"

$FANCY banner "jitLLM · LLM inference in pure Java on the GPU" "Java kernels JIT-compiled by TornadoVM, NVIDIA libraries where they win · RTX 4090"

if [ "$mode" = all ] || [ "$mode" = chat ]; then
    $FANCY act 1 "Generate" "$(basename "$MODEL_CHAT") · decode replayed as a CUDA graph · the tokens appear as they are generated"
    printf '    \033[1;36m❯\033[0m \033[1mIn two sentences: why run LLM inference in pure Java on a GPU?\033[0m\n\n    '
    # tokens stream to the terminal as they come; the JVM's warnings go to a log
    ./jitllm --gpu --cuda-graphs --model "$MODEL_CHAT" --max-new-tokens 120 \
        --prompt 'In two sentences: why run LLM inference in pure Java on a GPU?' 2>"$FANCY_LOGS/chat.err" \
        | tee "$FANCY_LOGS/chat.log" | sed -u 's/^/    /'
    # jitLLM prints its performance line on stderr
    $FANCY llm-chat "$FANCY_LOGS/chat.err"
    [ "$mode" = chat ] || fancy_pause
fi

if [ "$mode" = all ] || [ "$mode" = bench ]; then
    $FANCY act 2 "jitLLM vs llama.cpp" "$(basename "$MODEL_BENCH") · same GPU, same GGUF, same batch size   (slide 22)"
    spin "jitLLM prefill: cuBLAS GEMMs + Java kernels" "$FANCY_LOGS/pp.log" \
        ./jitllm bench --gpu --model "$MODEL_BENCH" --pp 512 --tg 0 --with-prefill-decode --batch-prefill-size 512 --with-native-libraries
    spin "jitLLM decode: Java kernels as a CUDA graph" "$FANCY_LOGS/tg.log" \
        ./jitllm bench --gpu --cuda-graphs --model "$MODEL_BENCH" --pp 0 --tg 128
    spin "llama.cpp: llama-bench pp512 + tg128" "$FANCY_LOGS/llama.log" \
        "$LLAMA_BENCH" -m "$MODEL_BENCH" -p 512 -n 128 -b 512 -ngl 99
    echo
    $FANCY llm-bench "$FANCY_LOGS/pp.log" "$FANCY_LOGS/tg.log" "$FANCY_LOGS/llama.log"
fi

$FANCY scoreboard "jitLLM on the RTX 4090"
