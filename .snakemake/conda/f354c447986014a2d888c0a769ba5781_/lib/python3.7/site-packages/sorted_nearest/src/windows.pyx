
from libc.stdint cimport int32_t

cimport cython

import numpy as np



def makewindows(indexes, starts, ends, window_size):

    _starts = starts - (starts % window_size)
    _ends = ends - (ends % window_size) + window_size

    max_n_windows = int((np.sum(_ends - _starts + window_size) / window_size))

    if starts.dtype == np.long:
        return makewindows64(indexes, starts, ends, max_n_windows, window_size)
    elif starts.dtype == np.int32:
        return makewindows32(indexes, starts, ends, max_n_windows, window_size)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(starts.dtype))


@cython.boundscheck(True)
@cython.wraparound(True)
@cython.initializedcheck(True)
cdef makewindows32(const long [::1] indexes, const int32_t [::1] starts, const int32_t [::1] ends, max_n_windows, int window_size):

    cdef:
        int nfound = 0
        int i = 0
        int length = len(starts)
        int start, end

    output_arr_indexes = np.ones(max_n_windows, dtype=long) * -1
    output_arr_start = np.ones(max_n_windows, dtype=np.int32) * -1
    output_arr_end = np.ones(max_n_windows, dtype=np.int32) * -1

    cdef long [::1] output_indexes
    cdef int32_t [::1] output_start
    cdef int32_t [::1] output_end

    output_indexes = output_arr_indexes
    output_start = output_arr_start
    output_end = output_arr_end

    for i in range(length):

        start = starts[i]
        end = ends[i]

        if start + window_size > end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            output_end[nfound] = end
            nfound += 1
            continue


        while start + window_size <= end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            start += window_size
            output_end[nfound] = start
            nfound += 1

        if start != end and start + window_size > end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            output_end[nfound] = end
            nfound += 1

    return output_arr_indexes[:nfound], output_arr_start[:nfound], output_arr_end[:nfound]



@cython.boundscheck(True)
@cython.wraparound(True)
@cython.initializedcheck(True)
cdef makewindows64(const long [::1] indexes, const long [::1] starts, const long [::1] ends, max_n_windows, int window_size):

    cdef:
        int nfound = 0
        int i = 0
        int length = len(starts)
        long start, end

    output_arr_indexes = np.ones(max_n_windows, dtype=long) * -1
    output_arr_start = np.ones(max_n_windows, dtype=long) * -1
    output_arr_end = np.ones(max_n_windows, dtype=long) * -1

    cdef long [::1] output_indexes
    cdef long [::1] output_start
    cdef long [::1] output_end

    output_indexes = output_arr_indexes
    output_start = output_arr_start
    output_end = output_arr_end

    for i in range(length):

        # print(i)
        start = starts[i]
        end = ends[i]

        if start + window_size > end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            output_end[nfound] = end
            nfound += 1
            continue


        while start + window_size <= end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            start += window_size
            output_end[nfound] = start
            nfound += 1

        if start != end and start + window_size > end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            output_end[nfound] = end
            nfound += 1

    return output_arr_indexes[:nfound], output_arr_start[:nfound], output_arr_end[:nfound]
