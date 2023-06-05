
from libc.stdint cimport int32_t

cimport cython

import numpy as np

def annotate_clusters(starts, ends, slack):

    if starts.dtype == np.long:
        return annotate_clusters64(starts, ends, slack)
    elif starts.dtype == np.int32:
        return annotate_clusters32(starts, ends, slack)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(starts.dtype))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef annotate_clusters64(const long [::1] starts, const long [::1] ends, int slack):

    cdef int min_start = starts[0]
    cdef int max_end = ends[0]
    cdef int i = 0
    cdef int n_clusters = 1
    cdef int length = len(starts)
   # cdef int last_write = 0

    output_arr_ids = np.ones(length, dtype=np.long) * -1

    cdef long [::1] output_ids

    output_ids = output_arr_ids

    for i in range(length):
        if not (starts[i] - slack) <= max_end:
            min_start = starts[i]
            max_end = ends[i]
            n_clusters += 1
            output_ids[i] = n_clusters
        else:
            output_ids[i] = n_clusters
            if ends[i] > max_end:
                max_end = ends[i]

    return output_arr_ids


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef annotate_clusters32(const int32_t [::1] starts, const int32_t [::1] ends, int slack):

    cdef int min_start = starts[0]
    cdef int max_end = ends[0]
    cdef int i = 0
    cdef int n_clusters = 1
    cdef int length = len(starts)

    output_arr_ids = np.ones(length, dtype=np.int32) * -1

    cdef int32_t [::1] output_ids

    output_ids = output_arr_ids

    for i in range(length):
        if not (starts[i] - slack) <= max_end:
            min_start = starts[i]
            max_end = ends[i]
            n_clusters += 1
            output_ids[i] = n_clusters
        else:
            output_ids[i] = n_clusters
            if ends[i] > max_end:
                max_end = ends[i]

    # if n_clusters != length + 1:
    #     output_arr_ids[i] = n_clusters

    return output_arr_ids

# from sorted_nearest import annotate_clusters
# import numpy as np
# starts = np.array([10, 20, 30])
# ends = np.array([15, 33, 35])
# annotate_clusters(starts, ends, 0)
