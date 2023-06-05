from libc.stdint cimport int32_t, int64_t

cimport cython

import numpy as np


def maketiles(indexes, starts, ends, tile_size, preserve_end=False):

    _starts = starts - (starts % tile_size)

    _ends_mod_zero = (ends % tile_size) == 0
    _ends = np.array(len(_starts), dtype=int)
    _ends = np.where(_ends_mod_zero,
                     (ends - 1) - ((ends - 1) % tile_size) + tile_size,
                     ends - (ends % tile_size) + tile_size)

    max_n_tiles = int((np.sum(_ends - _starts + tile_size) / tile_size))
    if max_n_tiles < 0:
        raise Exception("The sum of your chromosome lengths is below 0. Did you switch the start and end columns or try a negative tile size?")

    if starts.dtype == np.int64:
        return maketiles64(indexes, _starts, _ends, max_n_tiles, tile_size)
    elif starts.dtype == np.int32:
        return maketiles32(indexes, _starts, _ends, max_n_tiles, tile_size)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(starts.dtype))


@cython.boundscheck(True)
@cython.wraparound(True)
@cython.initializedcheck(True)
cdef maketiles32(const int64_t [::1] indexes, const int32_t [::1] starts, const int32_t [::1] ends, max_n_tiles, int tile_size):

    cdef:
        int nfound = 0
        int i = 0
        int length = len(starts)
        int32_t start, end

    output_arr_indexes = np.ones(max_n_tiles, dtype=np.int64) * -1
    output_arr_start = np.ones(max_n_tiles, dtype=np.int32) * -1
    output_arr_end = np.ones(max_n_tiles, dtype=np.int32) * -1

    cdef long [::1] output_indexes
    cdef int32_t [::1] output_start
    cdef int32_t [::1] output_end

    output_indexes = output_arr_indexes
    output_start = output_arr_start
    output_end = output_arr_end

    for i in range(length):

        start = starts[i]
        end = ends[i]

        if start + tile_size >= end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            output_end[nfound] = end
            nfound += 1
        else:
            while start + tile_size <= end:
                output_indexes[nfound] = indexes[i]
                output_start[nfound] = start
                start += tile_size
                output_end[nfound] = start
                nfound += 1

            if start > end:
                output_indexes[nfound] = indexes[i]
                output_start[nfound] = start
                output_end[nfound] = end
                nfound += 1

    return output_arr_indexes[:nfound], output_arr_start[:nfound], output_arr_end[:nfound]



@cython.boundscheck(True)
@cython.wraparound(True)
@cython.initializedcheck(True)
cdef maketiles64(const int64_t [::1] indexes, const int64_t [::1] starts, const int64_t [::1] ends, max_n_tiles, int tile_size):

    cdef:
        int nfound = 0
        int i = 0
        int length = len(starts)
        long start, end

    output_arr_indexes = np.ones(max_n_tiles, dtype=np.int64) * -1
    output_arr_start = np.ones(max_n_tiles, dtype=np.int64) * -1
    output_arr_end = np.ones(max_n_tiles, dtype=np.int64) * -1

    # return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
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

        if start + tile_size >= end:
            output_indexes[nfound] = indexes[i]
            output_start[nfound] = start
            output_end[nfound] = end
            nfound += 1
        else:
            while start + tile_size <= end:
                output_indexes[nfound] = indexes[i]
                output_start[nfound] = start
                start += tile_size
                output_end[nfound] = start
                nfound += 1

            if start > end:
                output_indexes[nfound] = indexes[i]
                output_start[nfound] = start
                output_end[nfound] = end
                nfound += 1

    return output_arr_indexes[:nfound], output_arr_start[:nfound], output_arr_end[:nfound]
