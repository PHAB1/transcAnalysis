"find clusters is what is called merge in pyranges"

from libc.stdint cimport int32_t

cimport cython

import numpy as np

def find_clusters(starts, ends, slack):

    if starts.dtype == np.long:
        return find_clusters64(starts, ends, slack)
    elif starts.dtype == np.int32:
        return find_clusters32(starts, ends, slack)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(starts.dtype))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef find_clusters64(const long [::1] starts, const long [::1] ends, int slack):

    cdef long min_start = starts[0]
    cdef long max_end = ends[0]
    cdef int i = 0
    cdef int intervals_in_cluster = 0
    cdef int n_clusters = 0
    cdef int length = len(starts)

    output_arr_start = np.ones(length, dtype=np.long) * -1
    output_arr_end = np.zeros(length, dtype=np.long) * -1
    output_arr_number = np.zeros(length, dtype=np.long) * -1

    cdef long [::1] output_start
    cdef long [::1] output_end
    cdef long [::1] output_number

    output_start = output_arr_start
    output_end = output_arr_end
    output_number = output_arr_number

    for i in range(length):
        if not (starts[i] - slack) <= max_end:
            output_start[n_clusters] = min_start
            output_end[n_clusters] = max_end
            output_number[n_clusters] = intervals_in_cluster
            min_start = starts[i]
            max_end = ends[i]
            intervals_in_cluster = 1
            n_clusters += 1
        else:
            intervals_in_cluster += 1
            if ends[i] > max_end:
                max_end = ends[i]

    if n_clusters < length and output_arr_start[n_clusters] != min_start:
        output_arr_start[n_clusters] = min_start
        output_arr_end[n_clusters] = max_end
        output_number[n_clusters] = intervals_in_cluster
        n_clusters += 1

    return output_arr_start[:n_clusters], output_arr_end[:n_clusters], output_arr_number[:n_clusters]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef find_clusters32(const int32_t [::1] starts, const int32_t [::1] ends, int slack):

    cdef int min_start = starts[0]
    cdef int max_end = ends[0]
    cdef int i = 0
    cdef int n_clusters = 0
    cdef int intervals_in_cluster = 0
    cdef int length = len(starts)

    output_arr_start = np.ones(length, dtype=np.int32) * -1
    output_arr_end = np.zeros(length, dtype=np.int32) * -1
    output_arr_number = np.zeros(length, dtype=np.int32) * -1

    cdef int32_t [::1] output_start
    cdef int32_t [::1] output_end
    cdef int32_t [::1] output_number

    output_start = output_arr_start
    output_end = output_arr_end
    output_number = output_arr_number

    for i in range(length):
        if not (starts[i] - slack) <= max_end:
            output_start[n_clusters] = min_start
            output_end[n_clusters] = max_end
            output_number[n_clusters] = intervals_in_cluster
            min_start = starts[i]
            max_end = ends[i]
            intervals_in_cluster = 1
            n_clusters += 1
        else:
            intervals_in_cluster += 1
            if ends[i] > max_end:
                max_end = ends[i]

    if n_clusters < length and output_arr_start[n_clusters] != min_start:
        output_arr_start[n_clusters] = min_start
        output_arr_end[n_clusters] = max_end
        output_number[n_clusters] = intervals_in_cluster
        n_clusters += 1

    return output_arr_start[:n_clusters], output_arr_end[:n_clusters], output_arr_number[:n_clusters]
