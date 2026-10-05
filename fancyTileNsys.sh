#!/usr/bin/env bash
# CUDA Tile from Java with NVIDIA Nsight Systems: fancyTile.sh, with the GEMM ladder profiled by nsys (~30 s).
#   1 threads, tiles and cuBLAS captured into one CUDA graph (demo 20), as in fancyTile.sh
#   2 the FP16 GEMM ladder (demo 25) run under `nsys profile --trace=cuda`, then drawn from nsys's own reports:
#     nsys's kernel-summary table (verbatim), kernel time and TFLOP/s per rung next to the program's wall clock,
#     and the GPU timeline of every launch from nsys's per-launch trace
#   bash fancyTileNsys.sh        NO_PAUSE=1 to skip the [enter] between acts; the nsys report stays in the log directory
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$DEMO_ROOT/fancy.sh"
use_tornadovm_7
BAILOUT="-Dtornado.recover.bailout=False"   # a tile kernel that fails to compile must not fall back to the CPU
command -v nsys >/dev/null || { echo "fancyTileNsys.sh: nsys is not on the PATH (CUDA toolkit bin/)" >&2; exit 1; }

$FANCY banner "CUDA Tile from Java · measured with NVIDIA Nsight Systems" "TornadoVM 7.0.0 · RTX 4090 · $(nsys --version | sed 's/NVIDIA Nsight Systems version /nsys /')"

$FANCY act 1 "Threads, tiles, cuBLAS: one CUDA graph" "a KernelContext kernel, a TileContext GEMM, cublasSgemv and a @Parallel loop, captured once   (slide 18)"
compile_demo 20-cutile-hybrid
(cd "$BUILD/20-cutile-hybrid" && spin "capturing and replaying the pipeline" "$FANCY_LOGS/pipeline.log" \
    tornado --printBytecodes --jvm="$BAILOUT" --classpath . TileHybridPipeline 256 2 graph)
$FANCY tile-pipeline "$FANCY_LOGS/pipeline.log"
fancy_pause

$FANCY act 2 "The GEMM ladder, as nsys sees the GPU" "FP16 GEMM, n = 2048: KernelContext, four tile shapes, a launch hint and cuBLAS, 10 calls each, under nsys   (slide 19)"
compile_demo 25-tile-ladder
[ -f "$TORNADOVM_HOME/tornado-argfile" ] || tornado --generate-argfile >/dev/null
printf '    \033[2m$ nsys profile --trace=cuda -o ladder java @tornado-argfile ... TileLadder 2048 10\033[0m\n'
(cd "$BUILD/25-tile-ladder" && spin "nsys profile: 8 rungs x 10 calls, every launch traced" "$FANCY_LOGS/program.log" \
    nsys profile --trace=cuda --force-overwrite=true -o "$FANCY_LOGS/ladder" \
    "$JAVA_HOME/bin/java" @"$TORNADOVM_HOME/tornado-argfile" "$BAILOUT" -cp . TileLadder 2048 10)
nsys_reports() {
    cd "$FANCY_LOGS"
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv -o kern ladder.nsys-rep
    nsys stats --force-export=true --report cuda_gpu_trace --format csv -o trace ladder.nsys-rep
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format column ladder.nsys-rep > kern-column.txt
}
spin "nsys stats: kernel summary + per-launch trace" "$FANCY_LOGS/nsys-stats.log" nsys_reports
echo
$FANCY nsys-raw "$FANCY_LOGS/kern-column.txt"
fancy_pause
$FANCY nsys-ladder "$FANCY_LOGS/kern_cuda_gpu_kern_sum.csv" "$FANCY_LOGS/trace_cuda_gpu_trace.csv" "$FANCY_LOGS/program.log" 2048
grep -q "All rungs produced the same, correct result" "$FANCY_LOGS/program.log" && ladder_ok=True || ladder_ok=False
python3 -c "import sys; sys.argv=['', '$FANCY_STATE']; open('$FANCY_STATE','a').write('Every rung correct\tvalidated by the program\t' + ('PASS' if $ladder_ok else 'FAIL') + '\tladder: every rung correct (under nsys too)\n')"
printf '    \033[2mnsys report: %s/ladder.nsys-rep (open it in the Nsight Systems GUI)\033[0m\n' "$FANCY_LOGS"

$FANCY scoreboard "CUDA Tile from Java, measured by nsys"
