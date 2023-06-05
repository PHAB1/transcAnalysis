#!/usr/bin/env python

from libc.stdint cimport int32_t

cimport cython

import numpy as np

def max_disjoint(indexes, starts, ends, slack):

    if starts.dtype == np.long:
        return max_disjoint64(indexes, starts, ends, slack)
    elif starts.dtype == np.int32:
        return max_disjoint32(indexes, starts, ends, slack)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(starts.dtype))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef max_disjoint64(const long [::1] indexes, const long [::1] starts, const long [::1] ends, int slack):

    cdef int length = len(starts)
    cdef int count = 0
    cdef int r1 = 0
    cdef int l1 = 0
    cdef int r2 = 0

    output_arr_indexes = np.ones(length, dtype=np.long) * -1
    # output_arr_start = np.ones(length, dtype=np.long) * -1
    # output_arr_end = np.zeros(length, dtype=np.long) * -1

    cdef long [::1] output_indexes
    # cdef long [::1] output_start
    # cdef long [::1] output_end

    output_indexes = output_arr_indexes
    # output_start = output_arr_start
    # output_end = output_arr_end

    # output_start[count] = starts[count]
    # output_end[count] = ends[count]
    output_indexes[count] = indexes[count]
    count += 1

    r1 = ends[0]
    for i in range(1, length):
        l1 = starts[i]
        r2 = ends[i]

        if l1 > (r1 + slack):
            # output_start[count] = l1
            # output_end[count] = r2
            output_indexes[count] = indexes[i]
            count += 1
            r1 = r2

    return output_arr_indexes[:count]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef max_disjoint32(const long [::1] indexes, const int32_t [::1] starts, const int32_t [::1] ends, int slack):

    cdef int length = len(starts)
    cdef int count = 0
    cdef int r1 = 0
    cdef int l1 = 0
    cdef int r2 = 0

    output_arr_indexes = np.ones(length, dtype=np.long) * -1
    # output_arr_start = np.ones(length, dtype=np.int32) * -1
    # output_arr_end = np.zeros(length, dtype=np.int32) * -1

    cdef long [::1] output_indexes
    # cdef int32_t [::1] output_start
    # cdef int32_t [::1] output_end

    output_indexes = output_arr_indexes
    # output_start = output_arr_start
    # output_end = output_arr_end

    # output_start[count] = starts[count]
    # output_end[count] = ends[count]
    output_indexes[count] = indexes[count]
    count += 1

    r1 = ends[0]
    for i in range(1, length):
        l1 = starts[i]
        r2 = ends[i]

        if l1 > (r1 + slack):
            # output_start[count] = l1
            # output_end[count] = r2
            output_indexes[count] = indexes[i]
            count += 1
            r1 = r2

    return output_arr_indexes[:count]
