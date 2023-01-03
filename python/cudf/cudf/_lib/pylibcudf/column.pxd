# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.types cimport size_type

from .column_view cimport ColumnView


cdef class Column:
    cdef unique_ptr[column] c_obj
    cdef column * get(self) nogil
    cpdef size_type size(self)
    cpdef size_type null_count(self)
    cpdef cbool has_nulls(self)
    cpdef ColumnView view(self)
    # cpdef data_type type(self)
    # cpdef column_view view()
    # cpdef mutable_column_view mutable_view()
    # cpdef column_contents release()
