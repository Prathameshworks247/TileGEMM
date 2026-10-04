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

## Why tiling wins

In the naive kernel each thread computes one element of `C` and reads a full row of
`A` and a full column of `B` from global memory. Across a block, the same values are
fetched over and over: every element of `B` is read by all N threads in its column.
Arithmetic is cheap and DRAM bandwidth is not, so the kernel is memory bound and the
SMs spend most of their time waiting.

The tiled kernel gives each block a `16x16` staging area in shared memory. Per step,
every thread loads exactly one element of `A` and one of `B` into that scratchpad,
the block synchronises, and then each thread reads 16 values back out of on-chip
memory to accumulate partial products. Each loaded value is used 16 times instead of
once, cutting global memory traffic by roughly the tile width and shifting the kernel
toward being compute bound.

## Results

Measured on a Tesla T4 (Colab), `nvcc -O3 -arch=sm_75`. GPU times are the mean of 5
runs after one warm-up; the CPU baseline is a single run. Times cover the multiply
only - host/device transfers are excluded.

<!-- Paste the ./matmul output here. The program prints these tables in markdown. -->

| N | CPU (ms) | Naive (ms) | Tiled (ms) | cuBLAS (ms) | Tiled GFLOPS | Tiled vs naive | Tiled % of cuBLAS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | | | | | | | |
| 1024 | | | | | | | |
| 2048 | | | | | | | |

All GPU results are verified against the CPU output to within a `1e-3` relative
tolerance; the benchmark exits non-zero if any implementation drifts past it.

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
