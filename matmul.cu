// TileGEMM - square matrix multiply four ways: CPU baseline, naive CUDA,
// shared-memory tiled CUDA, and cuBLAS as the ceiling to measure against.
//
// Build: make      Run: ./matmul

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE 16
#define RUNS 5

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(e_));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t s_ = (call);                                           \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__,      \
                    __LINE__, (int)s_);                                       \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ----------------------------------------------------------------- CPU baseline
// i,k,j order so the inner loop walks B and C with unit stride. The textbook
// i,j,k order strides B by N and runs several times slower, which would only
// flatter the GPU speedups.
static void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        float* crow = C + (size_t)i * N;
        for (int j = 0; j < N; ++j) crow[j] = 0.0f;
        for (int k = 0; k < N; ++k) {
            float a = A[(size_t)i * N + k];
            const float* brow = B + (size_t)k * N;
            for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
        }
    }
}

// ----------------------------------------------------------------- naive kernel
// One thread per output element; every operand is read straight from global
// memory. Each element of B is re-read by all N threads in its column - that
// wasted traffic is exactly what the tiled kernel recovers.
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < N; ++k) acc += A[(size_t)row * N + k] * B[(size_t)k * N + col];
    C[(size_t)row * N + col] = acc;
}

// ----------------------------------------------------------------- tiled kernel
// Each block stages a TILE x TILE block of A and of B into shared memory, so
// the two values a thread loads per step get reused TILE times from on-chip
// memory instead of being fetched from DRAM again.
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int ak = t * TILE + tx;  // column of A this thread loads
        int bk = t * TILE + ty;  // row of B this thread loads
        // Zero-fill out of range so an N that is not a multiple of TILE
        // contributes nothing to the dot product.
        As[ty][tx] = (row < N && ak < N) ? A[(size_t)row * N + ak] : 0.0f;
        Bs[ty][tx] = (bk < N && col < N) ? B[(size_t)bk * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();  // before the next step overwrites the tiles
    }

    if (row < N && col < N) C[(size_t)row * N + col] = acc;
}

// --------------------------------------------------------------------- harness

// One warm-up launch (JIT, cache fill, clock ramp), then RUNS timed launches
// bracketed by CUDA events. Returns mean ms per launch.
template <typename F>
static float time_gpu_ms(F launch) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < RUNS; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / RUNS;
}

// Relative, not absolute: a dot product of N random floats accumulates error
// proportional to the result's magnitude, so a fixed epsilon fails spuriously
// at N=2048 even when the kernel is correct.
static float max_rel_err(const float* ref, const float* got, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(got[i] - ref[i]) / fmaxf(fabsf(ref[i]), 1.0f);
        if (d > worst) worst = d;
    }
    return worst;
}

int main(void) {
    const int sizes[] = {512, 1024, 2048};
    const int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    printf("GPU: %s (sm_%d%d), %.1f GB, %d SMs\n", prop.name, prop.major,
           prop.minor, prop.totalGlobalMem / 1073741824.0,
           prop.multiProcessorCount);
    printf("TILE = %d. GPU times are the mean of %d runs after one warm-up; "
           "the CPU baseline is a single run.\n", TILE, RUNS);
    printf("Times cover the multiply only - host<->device copies are excluded.\n");

    for (int s = 0; s < nsizes; ++s) {
        int N = sizes[s];
        size_t elems = (size_t)N * N;
        size_t bytes = elems * sizeof(float);
        double flops = 2.0 * N * N * N;

        float* hA = (float*)malloc(bytes);
        float* hB = (float*)malloc(bytes);
        float* hRef = (float*)malloc(bytes);   // CPU result, the reference
        float* hGot = (float*)malloc(bytes);   // whichever GPU result we check
        if (!hA || !hB || !hRef || !hGot) { fprintf(stderr, "host alloc failed\n"); return 1; }

        srand(1234);  // same inputs every run and every size
        for (size_t i = 0; i < elems; ++i) {
            hA[i] = (float)rand() / RAND_MAX;
            hB[i] = (float)rand() / RAND_MAX;
        }

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

        printf("\n### N = %d\n\n", N);
        fflush(stdout);

        // --- CPU reference, timed once.
        auto t0 = std::chrono::steady_clock::now();
        matmul_cpu(hA, hB, hRef, N);
        auto t1 = std::chrono::steady_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        // --- naive
        float naive_ms = time_gpu_ms([&] { matmul_naive<<<grid, block>>>(dA, dB, dC, N); });
        CUDA_CHECK(cudaMemcpy(hGot, dC, bytes, cudaMemcpyDeviceToHost));
        float naive_err = max_rel_err(hRef, hGot, elems);

        // --- tiled
        float tiled_ms = time_gpu_ms([&] { matmul_tiled<<<grid, block>>>(dA, dB, dC, N); });
        CUDA_CHECK(cudaMemcpy(hGot, dC, bytes, cudaMemcpyDeviceToHost));
        float tiled_err = max_rel_err(hRef, hGot, elems);

        // --- cuBLAS. cuBLAS is column-major and our data is row-major. A
        // row-major MxK matrix is already a column-major KxM one, i.e. its
        // transpose, so asking for B^T * A^T with no transpose flags yields
        // (A*B)^T in column-major, which IS A*B read back as row-major.
        // Swapping the operands is the whole fix; adding OP_T here instead is
        // the classic way to get a silently transposed result.
        const float one = 1.0f, zero = 0.0f;
        float blas_ms = time_gpu_ms([&] {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &one, dB, N, dA, N, &zero, dC, N));
        });
        CUDA_CHECK(cudaMemcpy(hGot, dC, bytes, cudaMemcpyDeviceToHost));
        float blas_err = max_rel_err(hRef, hGot, elems);

        printf("| Implementation | Time (ms) | GFLOPS | vs CPU | vs naive | Max rel err |\n");
        printf("|---|---:|---:|---:|---:|---:|\n");
        printf("| CPU baseline | %.2f | %.2f | 1.00x | %.2fx | ref |\n",
               cpu_ms, flops / (cpu_ms * 1e6), naive_ms / cpu_ms);
        printf("| Naive CUDA | %.2f | %.2f | %.1fx | 1.00x | %.2e |\n",
               naive_ms, flops / (naive_ms * 1e6), cpu_ms / naive_ms, naive_err);
        printf("| Tiled CUDA (%dx%d) | %.2f | %.2f | %.1fx | %.2fx | %.2e |\n",
               TILE, TILE, tiled_ms, flops / (tiled_ms * 1e6),
               cpu_ms / tiled_ms, naive_ms / tiled_ms, tiled_err);
        printf("| cuBLAS | %.2f | %.2f | %.1fx | %.2fx | %.2e |\n",
               blas_ms, flops / (blas_ms * 1e6), cpu_ms / blas_ms,
               naive_ms / blas_ms, blas_err);
        printf("\nTiled reaches %.1f%% of cuBLAS.\n", 100.0 * blas_ms / tiled_ms);

        const float tol = 1e-3f;
        if (naive_err > tol || tiled_err > tol || blas_err > tol) {
            fprintf(stderr, "\nFAIL: a result exceeded the %.0e relative tolerance at N=%d\n", tol, N);
            return 1;
        }
        fflush(stdout);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hRef); free(hGot);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    printf("\nAll sizes verified against the CPU baseline.\n");
    return 0;
}
