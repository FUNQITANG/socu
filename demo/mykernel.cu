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

__global__ void syrk_update_kernel(const float* __restrict__ E,
                                  float* __restrict__ D,
                                  int B, int M, int n) {
    int bm = (int)blockIdx.x;
    int b = bm / M;
    int m = bm - b * M;
    int j = (int)threadIdx.x;
    int i = (int)threadIdx.y;
    // 这是 CUDA kernel 常见的“线程可能多于数据”的安全写法
    if (i >= n || j >= n) return;

    const float* E_mat = E + ((b * M + m) * n * n);
    float* D_mat = D + ((b * M + m) * n * n);

    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        acc += E_mat[i * n + k] * E_mat[j * n + k];
    }
    D_mat[i * n + j] -= acc;
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

// Synchronize a CUDA stream created by create_stream().
// Returns cudaError_t as int (0 == cudaSuccess).
DLL_EXPORT int __cdecl stream_synchronize(int64_t stream_handle) {
    cudaStream_t s = (cudaStream_t)stream_handle;
    return (int)cudaStreamSynchronize(s);
}

DLL_EXPORT void __cdecl syrk_update(const float* E, float* D, int B, int M, int n, int64_t stream_handle) {
/*
DLL_EXPORT void __cdecl launch_my_kernel(const float* in, float* out, int n, int64_t stream_handle) {
    cudaStream_t stream = (cudaStream_t)stream_handle;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    my_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}
*/
    cudaStream_t stream = (cudaStream_t)stream_handle;
    dim3 threads(16, 16, 1);
    dim3 blocks(B * M, 1, 1);
    syrk_update_kernel<<<blocks, threads, 0, stream>>>(E, D, B, M, n);
}
