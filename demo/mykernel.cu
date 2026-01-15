// mykernels.cu
#include <cuda_runtime.h>
#include <stdint.h>
/*
Python 要稳定调用 DLL，最重要的是导出函数的 ABI 必须“简单、稳定、可被 ctypes/cffi 绑定”。
0. “DLL 自己创建 stream 并把 handle 交回 Python”
1. 用 C ABI 导出（避免 C++ name mangling）
2. 只用基础类型/指针：void* / int32 / int64 / float / double
不要把 std::vector、at::Tensor、C++ class 之类跨边界。
3. 明确 calling convention（可选但建议）
ctypes 默认按 cdecl 调用。你最好显式声明为 __cdecl（MSVC 默认也是 cdecl，明确写能减少歧义）：
*/

// 用 C ABI 导出（避免 C++ name mangling）
#if defined(_WIN32)
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT extern "C"
#endif

// row-major 矩阵乘法更新 D -= E * E^T
__global__ void syrk_update_kernel(const float* __restrict__ E,
                                  float* __restrict__ D,
                                  int B, int M, int n) {
    const int bm = (int)blockIdx.z;     // block 对应第 bm 个 n x n 矩阵
    const int b = bm / M;
    const int m = bm - b * M;
    // int j = (int)threadIdx.x;
    // int i = (int)threadIdx.y;

    // 大矩阵 C 分块 GRID 计算时，blockIdx.x/y 用于区分块位置
    // TILE 的尺寸是 syrk_update() 调用 kernel 时指定的
    const int TILE_X = (int)blockDim.x;
    const int TILE_Y = (int)blockDim.y;
    int row = (int)blockIdx.y * TILE_Y + threadIdx.y;
    int col = (int)blockIdx.x * TILE_X + threadIdx.x;

    // 这是 CUDA kernel 常见的“线程可能多于数据”的安全写法
    // if (i >= n || j >= n) return;
    if (row >= n || col >= n) return;

    // size_t 64位无符号整数(x64)
    const size_t mat_stride = (size_t)n * (size_t)n;
    const size_t mat_offset = (size_t)(b * M + m) * mat_stride;
    const float* E_mat = E + mat_offset;
    float* D_mat = D + mat_offset;

    // Large-n 时 float32 逐项累加容易产生明显舍入误差；用 fp64 累加再转回 float32。
    const size_t row_offset = (size_t)row * (size_t)n;
    const size_t col_offset = (size_t)col * (size_t)n;

    double acc = 0.0;
    // float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        const double a = (double)E_mat[row_offset + (size_t)k];
        const double b2 = (double)E_mat[col_offset + (size_t)k];
        acc += a * b2;
    }

    D_mat[row_offset + (size_t)col] -= (float)acc;
}

DLL_EXPORT int64_t __cdecl create_stream() {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    return (int64_t)s;
}

DLL_EXPORT void __cdecl destroy_stream(int64_t stream_handle) {
    cudaStream_t s = (cudaStream_t)stream_handle;
    cudaStreamDestroy(s);
}
// ------------------------------
// Synchronize a CUDA stream created by create_stream().
// ------------------------------

// Returns cudaError_t as int (0 == cudaSuccess).
DLL_EXPORT int __cdecl stream_synchronize(int64_t stream_handle) {
    cudaStream_t s = (cudaStream_t)stream_handle;
    return (int)cudaStreamSynchronize(s);
}


// ------------------------------
// CUDA event timing helpers
// ------------------------------

// Create a CUDA event (timing-enabled). Returns an opaque handle.
DLL_EXPORT int64_t __cdecl create_event() {
    cudaEvent_t e;
    // Default events support timing.
    cudaEventCreateWithFlags(&e, cudaEventDefault);
    return (int64_t)(uintptr_t)e;
}

DLL_EXPORT void __cdecl destroy_event(int64_t event_handle) {
    cudaEvent_t e = (cudaEvent_t)(uintptr_t)event_handle;
    cudaEventDestroy(e);
}

// Record an event on a stream. Returns cudaError_t (0 == cudaSuccess).
DLL_EXPORT int __cdecl event_record(int64_t event_handle, int64_t stream_handle) {
    cudaEvent_t e = (cudaEvent_t)(uintptr_t)event_handle;
    cudaStream_t s = (cudaStream_t)stream_handle;
    return (int)cudaEventRecord(e, s);
}

// Synchronize a CUDA event. Returns cudaError_t (0 == cudaSuccess).
DLL_EXPORT int __cdecl event_synchronize(int64_t event_handle) {
    cudaEvent_t e = (cudaEvent_t)(uintptr_t)event_handle;
    return (int)cudaEventSynchronize(e);
}

// Returns elapsed time in milliseconds between two recorded events.
// If either event was not recorded, returns -1.
DLL_EXPORT float __cdecl event_elapsed_ms(int64_t start_event_handle, int64_t end_event_handle) {
    cudaEvent_t start = (cudaEvent_t)(uintptr_t)start_event_handle;
    cudaEvent_t end = (cudaEvent_t)(uintptr_t)end_event_handle;
    float ms = -1.0f;
    cudaError_t err = cudaEventElapsedTime(&ms, start, end);
    if (err != cudaSuccess) {
        return -1.0f;
    }
    return ms;
}


// ------------------------------
// syrk update wrapper
// ------------------------------
/*
DLL_EXPORT void __cdecl launch_my_kernel(const float* in, float* out, int n, int64_t stream_handle) {
    cudaStream_t stream = (cudaStream_t)stream_handle;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    my_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}
*/
DLL_EXPORT void __cdecl syrk_update(const float* E, float* D, int B, int M, int n, int64_t stream_handle) {
    /*
    * B * M 个 n x n 矩阵
    */
    cudaStream_t stream = (cudaStream_t)stream_handle;
    constexpr int TILE = 16;
    dim3 threads(TILE, TILE, 1);
    
    const int TILE_X = (int)threads.x;
    const int TILE_Y = (int)threads.y;
    int GRID_X = (n + TILE_X - 1) / TILE_X;
    int GRID_Y = (n + TILE_Y - 1) / TILE_Y;
    dim3 blocks(GRID_X, GRID_Y, B * M);
    syrk_update_kernel<<<blocks, threads, 0, stream>>>(E, D, B, M, n);
}
