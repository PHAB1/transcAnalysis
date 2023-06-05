

from libc.stdint cimport int32_t

cimport cython

import numpy as np

def cluster_by(starts, ends, ids, slack=0):

    if starts.dtype == np.long:
        return cluster_by64(starts, ends, ids.astype(np.long), slack)
    elif starts.dtype == np.int32:
        return cluster_by32(starts, ends, ids.astype(np.int32), slack)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(starts.dtype))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef cluster_by64(const long [::1] starts, const long [::1] ends, const long [::1] ids, int slack):

    cdef long min_start = starts[0]
    cdef long max_end = ends[0]
    cdef int i = 0
    cdef int current_id = ids[0]
    cdef int last_id = ids[0]
    cdef int n_clusters = 1
    cdef int length = len(starts)

    output_arr_ids = np.ones(length, dtype=np.long) * -1

    cdef long [::1] output_ids

    output_ids = output_arr_ids

    for i in range(length):
        current_id = ids[i]
        if (not (starts[i] - slack) <= max_end) or current_id != last_id:
            n_clusters += 1
            output_ids[i] = n_clusters
            min_start = starts[i]
            max_end = ends[i]
        else:
            output_ids[i] = n_clusters
            if ends[i] > max_end:
                max_end = ends[i]
        last_id = current_id

    # if n_clusters != length + 1:
    #     output_arr_ids[i] = n_clusters
    #     n_clusters += 1

    return output_arr_ids


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef cluster_by32(const int32_t [::1] starts, const int32_t [::1] ends, const int32_t [::1] ids, int slack):

    cdef int min_start = starts[0]
    cdef int max_end = ends[0]
    cdef int i = 0
    cdef int current_id = ids[0]
    cdef int last_id = ids[0]
    cdef int n_clusters = 1
    cdef int length = len(starts)

    output_arr_ids = np.ones(length, dtype=np.int32) * -1

    cdef int32_t [::1] output_ids

    output_ids = output_arr_ids

    for i in range(length):
        # print("i " * 5, i)
        current_id = ids[i]
        if (not (starts[i] - slack) <= max_end) or current_id != last_id:
            n_clusters += 1
            output_ids[i] = n_clusters
            min_start = starts[i]
            max_end = ends[i]
        else:
            # print("else")
            output_ids[i] = n_clusters
            if ends[i] > max_end:
                max_end = ends[i]
        last_id = current_id

    # if n_clusters != length + 1:
    #     output_arr_ids[i] = n_clusters
    #     n_clusters += 1

    return output_arr_ids
