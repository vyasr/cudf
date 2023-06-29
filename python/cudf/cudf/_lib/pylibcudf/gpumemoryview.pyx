# Copyright (c) 2023, NVIDIA CORPORATION.

from functools import cached_property


cdef class gpumemoryview:
    """Minimal representation of a memory buffer."""
    def __init__(self, object obj):
        try:
            cai = obj.__cuda_array_interface__
        except AttributeError:
            raise ValueError(
                "gpumemoryview must be constructed from an object supporting "
                "the CUDA array interface"
            )
        self.obj = obj
        # TODO: Need to respect readonly
        self.ptr = cai["data"][0]

    @cached_property
    def __cuda_array_interface__(self):
        return self.obj.__cuda_array_interface__
