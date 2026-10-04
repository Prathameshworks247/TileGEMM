# TileGEMM

Naive and shared-memory tiled CUDA kernels for matrix multiplication, benchmarked
against a CPU baseline and cuBLAS.

## What this is

This project explores how GPU memory hierarchy affects performance. It implements
matrix multiplication three ways: a CPU baseline, a naive CUDA kernel reading from
global memory, and a tiled kernel that loads 16x16 blocks into shared memory for
reuse. Each version is timed with CUDA events across matrix sizes from 512 to 2048,
reported in milliseconds and GFLOPS, and verified against the CPU output. cuBLAS is
included as a fourth measurement so the hand-written kernels have a realistic ceiling
to be judged against.

## How tiling works

In the naive kernel each thread computes one element of `C` and walks a full row of
`A` and a full column of `B`. Within a warp `threadIdx.x` varies fastest, so `col`
varies and the reads of `B` are already coalesced - the naive kernel is not the
pathological uncoalesced version you sometimes see. What it still does is re-fetch the
same values across the block: every element of `B` is read by all N threads in its
column, and every element of `A` by all N threads in its row.

The tiled kernel gives each block a `16x16` staging area in shared memory. Per step
every thread loads exactly one element of `A` and one of `B` into that scratchpad, the
block synchronises, and then each thread reads 16 values back out of on-chip memory to
accumulate partial products. Each loaded value is used 16 times instead of once, which
cuts requests to the memory system by roughly the tile width.

## Results

Measured on a Tesla T4 (sm_75, 40 SMs, 15 GB) on Colab, `nvcc -O3 -arch=sm_75`.
GPU times are the mean of 5 runs after one warm-up; the CPU baseline is a single run. Times cover the multiply
only - host/device transfers are excluded.

| N | CPU (ms) | Naive (ms) | Tiled (ms) | cuBLAS (ms) | Tiled GFLOPS | Tiled vs naive | Tiled vs CPU | Tiled % of cuBLAS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 23.6 | 0.67 | 0.44 | 0.08 | 616 | 1.54x | 54x | 18.9% |
| 1024 | 203.0 | 5.22 | 3.34 | 0.49 | 642 | 1.56x | 61x | 14.5% |
| 2048 | 1773.6 | 48.06 | 25.20 | 2.90 | 682 | 1.91x | 70x | 11.5% |

cuBLAS peaks at 5933 GFLOPS at N=2048, about 73% of the T4's ~8.1 TFLOPS fp32
ceiling. The tiled kernel's 682 GFLOPS is ~8% of that ceiling.

### Reading these numbers honestly

Tiling buys **1.9x over naive at N=2048**, not the 10x the "16x less traffic" argument
would suggest. Two reasons, and both are the actual lesson of the project:

- The naive kernel's loads are coalesced and its working set hits the T4's 4 MB L2, so
  it was never paying full DRAM latency on most accesses. Tiling moves traffic from L2
  to shared memory, a smaller win than DRAM-to-shared would have been.
- At `16x16`, each thread computes a single output and the inner loop is two shared
  loads per fused multiply-add. That ratio, not memory bandwidth, is now the limit -
  the kernel is bound by shared-memory throughput and loop overhead rather than DRAM.

That is also why the gap to cuBLAS stays large: cuBLAS gets its remaining ~9x from
register blocking (each thread computing a patch of `C`, so loaded values are reused
out of registers rather than re-read from shared memory), double-buffered loads, and
tuning per shape. The honest summary is that ~60 lines of tiled CUDA gets within an
order of magnitude of a vendor library, and the next 10x needs register blocking.

Note the speedup grows with N (1.54x to 1.91x) - the larger the matrix, the less of it
fits in cache and the more the staging pays off.

All GPU results are verified against the CPU output to within a `1e-3` relative
tolerance - worst observed was `3.4e-06`. The benchmark exits non-zero if any
implementation drifts past it.

## Running it

Needs an NVIDIA GPU and the CUDA toolkit. If you do not have one locally, open
`run_colab.ipynb` in Google Colab, set the runtime to a T4 GPU, and run the cells.

```sh
make            # defaults to -arch=sm_75 (T4); override with make ARCH=sm_80
./matmul
```

## Files

| File | |
|---|---|
| `matmul.cu` | All four implementations, the CUDA-event timing harness, and verification |
| `Makefile` | One `nvcc` invocation; `ARCH` is overridable |
| `run_colab.ipynb` | Launcher for running the benchmark on a free Colab T4 |

## Notes on the implementation

- **Loop order in the CPU baseline** is `i,k,j`, not the textbook `i,j,k`. The inner
  loop then walks `B` and `C` with unit stride instead of striding `B` by `N`, which
  is several times faster. Benchmarking against the slower ordering would have
  inflated every GPU speedup for free.
- **cuBLAS is column-major**, this code is row-major. `cublasSgemm` is called with the
  operands swapped and no transpose flags, which produces the right answer without
  any explicit transposes - see the comment in `matmul.cu`. Reaching for `CUBLAS_OP_T`
  instead is the usual way to end up with a silently transposed result.
- **Verification uses relative error.** A dot product over 2048 floats accumulates
  rounding error proportional to the magnitude of the result, so a fixed absolute
  epsilon reports failures at N=2048 even for a correct kernel.
- **The tiled kernel guards its loads** and zero-fills out of range, so sizes that are
  not multiples of 16 still give correct results.

## Possible next steps

Register blocking (each thread computing a 4x4 patch of `C` in registers) is the
standard next step and closes much of the remaining gap to cuBLAS. Beyond that:
double-buffered tile loads to overlap the global loads with the math, padding shared
memory to avoid bank conflicts, and tensor cores for fp16/tf32.
