#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>

// These functions are implemented in mykernel.cu and exported from the mykernel shared library.
extern "C" {
int64_t __cdecl create_stream();
void __cdecl destroy_stream(int64_t stream_handle);
int __cdecl stream_synchronize(int64_t stream_handle);
void __cdecl syrk_update(const float* E, float* D, int B, int M, int n, int64_t stream_handle);
}

static PyObject* py_create_stream(PyObject*, PyObject*) {
    int64_t stream = create_stream();
    return PyLong_FromLongLong((long long)stream);
}

static PyObject* py_destroy_stream(PyObject*, PyObject* args) {
    long long stream_handle = 0;
    if (!PyArg_ParseTuple(args, "L", &stream_handle)) {
        return nullptr;
    }
    destroy_stream((int64_t)stream_handle);
    Py_RETURN_NONE;
}

static PyObject* py_stream_synchronize(PyObject*, PyObject* args) {
    long long stream_handle = 0;
    if (!PyArg_ParseTuple(args, "L", &stream_handle)) {
        return nullptr;
    }
    int err = stream_synchronize((int64_t)stream_handle);
    return PyLong_FromLong((long)err);
}

static PyObject* py_syrk_update(PyObject*, PyObject* args) {
    unsigned long long E_ptr = 0;
    unsigned long long D_ptr = 0;
    int B = 0;
    int M = 0;
    int n = 0;
    long long stream_handle = 0;

    // Pointers are passed in as Python ints (device pointers from Warp/CuPy/Torch).
    if (!PyArg_ParseTuple(args, "KKiiiL", &E_ptr, &D_ptr, &B, &M, &n, &stream_handle)) {
        return nullptr;
    }

    syrk_update((const float*)(uintptr_t)E_ptr, (float*)(uintptr_t)D_ptr, B, M, n, (int64_t)stream_handle);
    Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"create_stream", (PyCFunction)py_create_stream, METH_NOARGS, "Create a non-blocking CUDA stream and return its handle."},
    {"destroy_stream", (PyCFunction)py_destroy_stream, METH_VARARGS, "Destroy a CUDA stream previously created by create_stream()."},
    {"stream_synchronize", (PyCFunction)py_stream_synchronize, METH_VARARGS, "Synchronize a CUDA stream. Returns cudaError_t (0 means success)."},
    {"syrk_update", (PyCFunction)py_syrk_update, METH_VARARGS, "Launch syrk_update kernel: syrk_update(E_ptr, D_ptr, B, M, n, stream_handle)."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "_mykernel",
    "CPython extension wrapper for the socu demo CUDA kernel.",
    -1,
    Methods,
};

PyMODINIT_FUNC PyInit__mykernel(void) {
    return PyModule_Create(&ModuleDef);
}
