from libc.stdint cimport int32_t

cimport cython

import numpy as np

def nearest_nonoverlapping(prev_ridx, prev_dist,
                           next_ridx, next_dist):

    if prev_dist.dtype == np.int32:
        return nearest_nonoverlapping32(prev_ridx, prev_dist,
                                        next_ridx, next_dist)
    elif prev_dist.dtype == np.int64:
        return nearest_nonoverlapping64(prev_ridx, prev_dist,
                                        next_ridx, next_dist)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(prev_dist.dtype))

def nearest_previous_nonoverlapping(ls, re, ridx):

    if ls.dtype == np.int32:
        return nearest_previous_nonoverlapping32(ls, re, ridx)
    elif ls.dtype == np.int64:
        return nearest_previous_nonoverlapping64(ls, re, ridx)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(ls.dtype))


def nearest_next_nonoverlapping(le, rs, ridx):

    if le.dtype == np.int32:
        return nearest_next_nonoverlapping32(le, rs, ridx)
    elif le.dtype == np.int64:
        return nearest_next_nonoverlapping64(le, rs, ridx)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(le.dtype))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_next_nonoverlapping64(const long [::1] l_e, const long [::1] r_s, const long [::1] r_idx):

    cdef int j = 0
    cdef int i = 0

    cdef int len_l = len(l_e)
    cdef int len_r = len(r_s)

    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.long) * -1
    cdef long [::1] ridx
    cdef long [::1] dist

    ridx = arr_ridx
    dist = arr_dist

    # print("l_e", list(l_e))
    # print("r_s", list(r_s))
    while i < len_l and j < len_r:
        # print("l_e[i]", l_e[i])
        # print("r_s[j]", r_s[j])
        if l_e[i] >= r_s[j]:
            # print("in if")
            j += 1
        else:
            # print("in else")
            dist[i] = r_s[j] - l_e[i]
            ridx[i] = r_idx[j]
            i += 1
        # print("dist", list(dist))
        # print("ridx", list(ridx))

    return arr_ridx, arr_dist

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_next_nonoverlapping32(const int32_t [::1] l_e, const int32_t [::1] r_s, const long [::1] r_idx):

    cdef int j = 0
    cdef int i = 0

    cdef int len_l = len(l_e)
    cdef int len_r = len(r_s)

    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    ridx = arr_ridx
    dist = arr_dist

    # print("l_e", list(l_e))
    # print("r_s", list(r_s))
    while i < len_l and j < len_r:
        # print("l_e[i]", l_e[i])
        # print("r_s[j]", r_s[j])
        if l_e[i] >= r_s[j]:
            # print("in if")
            j += 1
        else:
            # print("in else")
            dist[i] = r_s[j] - l_e[i]
            ridx[i] = r_idx[j]
            i += 1
            # print("dist", list(dist))
            # print("ridx", list(ridx))

    return arr_ridx, arr_dist



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_previous_nonoverlapping64(const long [::1] l_s, const long [::1] r_e, const long [::1] r_idx):

    cdef int len_l = len(l_s)
    cdef int len_r = len(r_e)

    cdef int j = len_r - 1
    cdef int i = len_l - 1

    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.long) * -1
    cdef long [::1] ridx
    cdef long [::1] dist

    ridx = arr_ridx
    dist = arr_dist

    while -1 < i and -1 < j:
        if l_s[i] <= r_e[j]:
            # print("in if")
            j -= 1
        else:
            # print("in else")
            # print("l_s[i]", l_s[i])
            # print("r_e[j]", r_e[j])
            dist[i] = l_s[i] - r_e[j]
            ridx[i] = r_idx[j]
            i -= 1
        # print("dist", list(dist))
        # print("ridx", list(ridx))

    # print("final dist", list(dist))
    # print("final ridx", list(ridx))

    return arr_ridx, arr_dist

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_previous_nonoverlapping32(const int32_t [::1] l_s, const int32_t [::1] r_e, const long [::1] r_idx):

    cdef int len_l = len(l_s)
    cdef int len_r = len(r_e)

    cdef int j = len_r - 1
    cdef int i = len_l - 1

    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    ridx = arr_ridx
    dist = arr_dist

    while -1 < i and -1 < j:
        if l_s[i] <= r_e[j]:
            # print("in if")
            j -= 1
        else:
            # print("in else")
            # print("l_s[i]", l_s[i])
            # print("r_e[j]", r_e[j])
            dist[i] = l_s[i] - r_e[j]
            ridx[i] = r_idx[j]
            i -= 1
            # print("dist", list(dist))
            # print("ridx", list(ridx))

    # print("final dist", list(dist))
    # print("final ridx", list(ridx))

    return arr_ridx, arr_dist

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_nonoverlapping64(const long [::1] prev_ridx, const long [::1] prev_dist,
                               const long [::1] next_ridx, const long [::1] next_dist):

    cdef int i = 0

    cdef int length = len(prev_ridx)

    output_arr_ridx = np.ones(length, dtype=np.long) * -1
    output_arr_dist = np.ones(length, dtype=np.long) * -1

    cdef long [::1] output_ridx
    cdef long [::1] output_dist

    output_ridx = output_arr_ridx
    output_dist = output_arr_dist

    for i in range(length):
        if next_ridx[i] == -1 and prev_ridx[i] > -1:
            output_ridx[i] = prev_ridx[i]
            output_dist[i] = prev_dist[i]
        elif prev_ridx[i] > -1 and next_ridx[i] > -1 and prev_dist[i] <= next_dist[i]:
            output_ridx[i] = prev_ridx[i]
            output_dist[i] = prev_dist[i]
        elif next_dist[i] > -1:
            output_ridx[i] = next_ridx[i]
            output_dist[i] = next_dist[i]

    return output_arr_ridx, output_arr_dist


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_nonoverlapping32(const long [::1] prev_ridx, const int32_t [::1] prev_dist,
                               const long [::1] next_ridx, const int32_t [::1] next_dist):

    cdef int i = 0

    cdef int length = len(prev_ridx)

    output_arr_ridx = np.ones(length, dtype=np.long) * -1
    output_arr_dist = np.ones(length, dtype=np.int32) * -1

    cdef long [::1] output_ridx
    cdef int32_t [::1] output_dist

    output_ridx = output_arr_ridx
    output_dist = output_arr_dist

    for i in range(length):
        if next_ridx[i] == -1 and prev_ridx[i] > -1:
            output_ridx[i] = prev_ridx[i]
            output_dist[i] = prev_dist[i]
        elif prev_ridx[i] > -1 and next_ridx[i] > -1 and prev_dist[i] <= next_dist[i]:
            output_ridx[i] = prev_ridx[i]
            output_dist[i] = prev_dist[i]
        elif next_dist[i] > -1:
            output_ridx[i] = next_ridx[i]
            output_dist[i] = next_dist[i]

    return output_arr_ridx, output_arr_dist
