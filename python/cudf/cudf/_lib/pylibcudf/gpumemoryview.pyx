# Copyright (c) 2023, NVIDIA CORPORATION.

from rmm._lib.device_buffer cimport DeviceBuffer


cdef class gpumemoryview:
    """Minimal representation of a memory buffer."""
    def __init__(self, Py_ssize_t ptr):
        # TODO: I don't think this constructor makes sense. The object should
        # only ever be constructible from some other object that owns memory
        self.ptr = ptr
        self.base = None


# TODO: Eventually cpdef classmethod
# TODO: Support arbitrary CAI objects
cpdef gpumemoryview_from_device_buffer(DeviceBuffer buf):
    cdef gpumemoryview ret = gpumemoryview.__new__(gpumemoryview)
    ret.base = buf
    ret.ptr = buf.ptr
    return ret
