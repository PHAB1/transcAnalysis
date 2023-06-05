from libc.stdint cimport int32_t, int64_t

cimport cython

import numpy as np


def k_nearest_previous_nonoverlapping(ls, re, lidx, ridx, k, ties=None):

    # print("ls", ls)
    # print("re", re)
    # print("lidx", lidx)
    # print("ridx", ridx)

    if ls.dtype == np.int32 and not ties:
        return nearest_previous_nonoverlapping32_all(ls, re, lidx, ridx, k)
    elif ls.dtype == np.int32 and ties == "first":
        return nearest_previous_nonoverlapping32_first(ls, re, lidx, ridx, k)
    elif ls.dtype == np.int32 and ties == "last":
        return nearest_previous_nonoverlapping32_last(ls, re, lidx, ridx, k)
    elif ls.dtype == np.int32 and ties == "different":
        res = nearest_previous_nonoverlapping32_k_distances(ls, re, lidx, ridx, k)
        return res
    elif ls.dtype == np.int64 and not ties:
        return nearest_previous_nonoverlapping64_all(ls, re, lidx, ridx, k)
    elif ls.dtype == np.int64 and ties == "first":
        return nearest_previous_nonoverlapping64_first(ls, re, lidx, ridx, k)
    elif ls.dtype == np.int64 and ties == "last":
        return nearest_previous_nonoverlapping64_last(ls, re, lidx, ridx, k)
    elif ls.dtype == np.int64 and ties == "different":
        res = nearest_previous_nonoverlapping64_k_distances(ls, re, lidx, ridx, k)
        return res
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(ls.dtype))

def k_nearest_next_nonoverlapping(le, rs, lidx, ridx, k, ties=None):

    # print("dtype and ties")
    # print(le.dtype)
    # print(ties)

    if le.dtype == np.int32 and not ties:
        return nearest_next_nonoverlapping32_all(le, rs, lidx, ridx, k)
    elif le.dtype == np.int32 and ties == "first":
        return nearest_next_nonoverlapping32_first(le, rs, lidx, ridx, k)
    elif le.dtype == np.int32 and ties == "last":
        return nearest_next_nonoverlapping32_last(le, rs, lidx, ridx, k)
    elif le.dtype == np.int32 and ties == "different":
        res = nearest_next_nonoverlapping32_k_distances(le, rs, lidx, ridx, k)
        return res
    elif le.dtype == np.int64 and not ties:
        return nearest_next_nonoverlapping64_all(le, rs, lidx, ridx, k)
    elif le.dtype == np.int64 and ties == "first":
        return nearest_next_nonoverlapping64_first(le, rs, lidx, ridx, k)
    elif le.dtype == np.int64 and ties == "last":
        return nearest_next_nonoverlapping64_last(le, rs, lidx, ridx, k)
    elif le.dtype == np.int64 and ties == "different":
        res = nearest_next_nonoverlapping64_k_distances(le, rs, lidx, ridx, k)
        return res
    else:
        raise Exception("Starts/Ends not int64 or int64: " + str(le.dtype))



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_previous_nonoverlapping32_all(const int32_t [::1] l_s, const int32_t [::1] r_e,
                                            const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_s)
    # cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int j = 0 # counter for r_e
    cdef int i = 0 # counter for l_s
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int _dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    for i in range(len_l):

        # print("-----" * 10)
        # print("i", i)

        h = r_idx[i]
        nfound_local = 0
        last = r_e[h]

        while h >= 0 and nfound_local < k[i]:

            # print("h", h, "nfound_local", nfound_local)
            if nfound == outarr_length - 1:

                # print("-- resizing --" * 5)

                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                # print("after resizing", arr_lidx)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = l_s[i] - r_e[h]

            # print(arr_lidx)
            # print(arr_ridx)
            # print(arr_dist)

            nfound += 1

            # print("last was", last, "r_e[h] is", r_e[h])
            if r_e[h] != last:
                nfound_local += 1

            last = r_e[h]
            h -= 1

        # print("while done. h:", h, "nfound_local:", nfound_local)

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_next_nonoverlapping32_all(const int32_t [::1] l_e, const int32_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int _dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    for i in range(len_l):

        h = r_idx[i]
        nfound_local = 0
        last = r_s[h]
        # print("-----" * 10)
        # print("i", i)

        while h < len_r and nfound_local < k[i]:

            # print("h", h, "nfound_local", nfound_local)

            if nfound == outarr_length - 1:
                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = r_s[h] - l_e[i]

            # print(arr_lidx)
            # print(arr_ridx)
            # print(arr_dist)

            nfound += 1

            # print("last was", last, "r_s[h] is", r_s[h])
            if r_s[h] != last:
                nfound_local += 1

            last = r_s[h]
            h += 1

        # print("while done. h:", h, "nfound_local:", nfound_local)
        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_previous_nonoverlapping32_k_distances(const int32_t [::1] l_s, const int32_t [::1] r_e,
                                                    const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    # print("nearest_previous_nonoverlapping32_k_distances " * 5)
    cdef int len_l = len(l_s)
    # cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int j = 0 # counter for r_e
    cdef int i = 0 # counter for l_s
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int last_dist = -1
    cdef int curr_dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    while i < len_l:

        h = r_idx[i]
        nfound_local = 0
        last_dist = -1

        while h >= 0 and nfound_local < k[i]:

            if nfound == outarr_length - 1:

                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = l_s[i] - r_e[h]
            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist

            nfound += 1

            if curr_dist != last_dist:
                nfound_local += 1

            curr_dist = last_dist

            last = r_e[h]
            h -= 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_next_nonoverlapping32_k_distances(const int32_t [::1] l_e, const int32_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int curr_dist = -1
    cdef int last_dist = -1


    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    while i < len_l:

        h = r_idx[i]
        nfound_local = 0
        last = -1

        while h < len_r and nfound_local < k[i]:

            if nfound == outarr_length - 1:

                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist


            curr_dist = r_s[h] - l_e[i]

            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist

            nfound += 1

            if curr_dist != last_dist:
                nfound_local += 1

            last_dist = curr_dist

            last = r_s[h]
            h += 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_next_nonoverlapping32_first(const int32_t [::1] l_e, const int32_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    # print("nearest_next_nonoverlapping first")

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int curr_dist = -1
    cdef int last_dist = -1


    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    # print("ridx", list(r_idx))
    # print("lidx", list(l_idx))
    # print("l_e", list(l_e))
    # print("r_s", list(r_s))
    # print("i", i)
    # print("len_l", len_l)
    # print("len_rx", len_rx)
    # print("j", j)
    while i < len_l:

        nfound_local = 0
        last_dist = -1

        h = r_idx[i]

        while h < len_r and nfound_local < k[i]:

            # print("h", h)
            # print("rs", r_s[h])
            # print("le", l_e[i])

            if nfound >= outarr_length - 1:


                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = r_s[h] - l_e[i]

            if curr_dist != last_dist:
                # print("inserting ridx", h, "lidx", l_idx[i], "dist", curr_dist)
                ridx[nfound] = h
                lidx[nfound] = l_idx[i]
                dist[nfound] = curr_dist

                nfound += 1

                nfound_local += 1

                last_dist = curr_dist

            last = r_s[h]
            h += 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_previous_nonoverlapping32_first(const int32_t [::1] l_s, const int32_t [::1] r_e,
                                              const long [::1] l_idx, const long [::1] r_idx,
                                              const long [::1] k):

    # print("nearest previous nonoverlapping first")

    cdef int len_l = len(l_s)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int j = 0 # counter for r_e
    cdef int i = 0 # counter for l_s
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int last_dist = -1
    cdef int curr_dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int32) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int32_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    # print("r_idx", list(r_idx))
    # print("r_e", list(r_e))
    # print("k", list(k))

    while i < len_l:

        last_dist = -1
        nfound_local = 0
        h = r_idx[i]

        while nfound_local < k[i] and h >= 0:  # and l_s[i] - r_e[h] >= 0:

            # print("h", h)
            # print("ls", l_s[i])
            # print("re", r_e[h])

            if nfound >= outarr_length - 1:
                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = l_s[i] - r_e[h]

            if curr_dist != last_dist:
                # print("inserting ridx", h, "lidx", l_idx[i], "dist", curr_dist)
                ridx[nfound] = h
                lidx[nfound] = l_idx[i]
                dist[nfound] = curr_dist
                last_dist = curr_dist

                nfound += 1

                nfound_local += 1

            last = r_e[h]
            h -= 1
            # print("h", h, "len_rx", len_rx)

        i += 1
        # print("i", i, "len_l", len_l)

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_next_nonoverlapping32_last(const int32_t [::1] l_e, const int32_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    # print("  nearest_next_nonoverlapping last" * 5)

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = 0
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int curr_dist = -1
    cdef int last_dist = -1
    cdef int start_dist = -1


    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int64) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef long [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    for i in range(len_l):

        # print("i", i)
        nfound_local = 0
        h = r_idx[i]
        last_dist = r_s[h] - l_e[i]
        start_dist = last_dist

        while h < len_r and nfound_local < k[i]:

            # print("h", h)
            # print("rs", r_s[h])
            # print("le", l_e[i])

            if nfound >= outarr_length - 1:

                # print("--resizing--" * 5)
                # print("before resizing", arr_lidx)

                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                # print("after resizing", arr_lidx)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = r_s[h] - l_e[i]

            """Cases:
1. all dist per i equal. need to add last dist manually
2. dists unequal -> all will be added but last. Need to add last manually
"""
            # if unequal, begin searching for last for next distance
            if curr_dist != last_dist:
                # print("if unequal, begin searching for last for next distance")
                nfound += 1
                nfound_local += 1

            # for each entry, add distance
            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist


            # print("lidx", list(lidx))
            # print("ridx", list(ridx))
            # print("dist", list(dist))

            last_dist = curr_dist

            h += 1

        # in case of all equal, need to add manually
        if start_dist == last_dist and k[i]:
            # print("in case of all equal, need to add manually")
            ridx[nfound] = h - 1
            lidx[nfound] = l_idx[i]
            dist[nfound] = last_dist
            # print("lidx", list(lidx))
            # print("ridx", list(ridx))
            # print("dist", list(dist))
            nfound += 1


        # in case last dist was not added
        elif nfound_local < k[i] and dist[nfound - 1] != last_dist:
            # print("FINALLY inserting lidx", l_idx[i], "ridx", h - 1, "dist", curr_dist)

            ridx[nfound] = h - 1
            lidx[nfound] = l_idx[i]
            dist[nfound] = last_dist
            nfound += 1
            # print("lidx", list(lidx))
            # print("ridx", list(ridx))
            # print("dist", list(dist))

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_previous_nonoverlapping32_last(const int32_t [::1] l_s, const int32_t [::1] r_e,
                                              const long [::1] l_idx, const long [::1] r_idx,
                                              const long [::1] k):

    # print("  nearest previous nonoverlapping last" * 5)

    cdef int len_l = len(l_s)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int i = 0 # counter for r_e
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = -1
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int last_dist = -1
    cdef int curr_dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int64) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef long [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    # print("l_idx", list(l_idx))
    # print("r_idx", list(r_idx))
    # print("r_e", list(r_e))
    # print("k", list(k))
    # print("len_l", len_l)

    for i in range(len_l):

        nfound_local = 0
        h = r_idx[i]
        last_dist = -1

        # print("----" * 5)
        # print("i", i)

        while nfound_local < k[i] and h >= 0:  # and l_s[i] - r_e[h] >= 0:

            # print("***" * 3)
            # print("h", h)
            # print("ls", l_s[i])
            # print("re", r_e[h])

            if nfound >= outarr_length - 1:

                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                # print("after resizing", arr_lidx)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = l_s[i] - r_e[h]

            # print("inserting lidx", l_idx[i], "ridx", h, "dist", curr_dist, "i", i)
            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist

            # print("curr_dist", curr_dist, "vs last_dist", last_dist)
            if curr_dist != last_dist:
                nfound += 1
                nfound_local += 1
                # print("ridx", list(ridx))
                # print("nfound", nfound)
            last_dist = curr_dist

            # print("lidx", list(lidx))
            # print("ridx", list(ridx))
            # print("dist", list(dist))

            h -= 1

        if nfound > 0 and nfound_local < k[i] and dist[nfound - 1] != curr_dist:

            # print("FINALLY inserting lidx", l_idx[i], "ridx", h + 1, "dist", curr_dist)
            ridx[nfound] = h + 1
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist
            nfound += 1

            # print("lidx", list(lidx)[:nfound])
            # print("ridx", list(ridx)[:nfound])
            # print("dist", list(dist)[:nfound])

        i += 1
        # print("i", i, "len_l", len_l)

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef get_all_ties(const int64_t[::1] lx, const int64_t[::1] ids, const int64_t[::1] dist, int k):

    # print("lx", list(lx))
    # print("ids", list(ids))
    # print("dist", list(dist))

    # for each index
    # want to keep until we have k
    # then keep all with same distance
    cdef:
        int i = 0
        int first_i = 0
        int current_found = 0
        int nfound = 0
        int current_id = ids[0]
        int last_id = -1
        int length = len(lx)
        int current_dist = -1
        int last_dist = -1

    arr_lidx = np.ones(length, dtype=np.long) * -1

    cdef long [::1] lidx

    lidx = arr_lidx

    # Chromosome  Start  End Name  Score Strand  __k__  __IX__  Start_b  End_b Name_b  Score_b Strand_b  Distance
    # 0       chr1      1    2    a      0      -      2       0        1      2      a        0        -         0
    # 1       chr1      1    2    a      0      -      2       0        1      2      a        0        -         0
    # 2       chr1      1    2    a      0      -      2       0        1      2      a        0        -         0

    while i < length:

        # print("i", i)
        # print("ids[i]", ids[i])
        # print("last_id", last_id)
        # print("current_found", current_found)
        # print("nfound", nfound)
        # print("length", length)
        first_i = i

        if ids[i] != last_id:
            # we have come to a new id, need to start counting again


            current_found = 0
            current_id = ids[i]
            last_id = ids[i]
            continue

        # while have not found k different ties, continue
        while i < length and current_found < k:
            # print("**first while**")
            # print("i", i, "length", length, "ids[i]", ids[i], "last_id", last_id, "dist[i]", last_dist, "current_found", current_found)
            lidx[nfound] = lx[i]
            # print("lidx", arr_lidx)
            last_dist = dist[i]
            nfound += 1
            current_found += 1
            i += 1

        # while we have not found all ties with same length as last included
        while i < length and ids[i] == last_id and dist[i] == last_dist:
            # print("**second while**")
            # print("i", i, "length", length, "ids[i]", ids[i], "last_id", last_id, "dist[i]", last_dist, "current_found", current_found)
            lidx[nfound] = lx[i]
            # print("lidx", arr_lidx)
            nfound += 1
            i += 1

        if i == first_i:
            i += 1

        # print(arr_lidx)

    return arr_lidx[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_previous_nonoverlapping64_all(const int64_t [::1] l_s, const int64_t [::1] r_e,
                                            const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_s)
    # cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int j = 0 # counter for r_e
    cdef int i = 0 # counter for l_s
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int _dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int64) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int64_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    for i in range(len_l):

        # print("-----" * 10)
        # print("i", i)

        h = r_idx[i]
        nfound_local = 0
        last = r_e[h]

        while h >= 0 and nfound_local < k[i]:

            # print("h", h, "nfound_local", nfound_local)
            if nfound == outarr_length - 1:

                # print("-- resizing --" * 5)

                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                # print("after resizing", arr_lidx)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = l_s[i] - r_e[h]

            # print(arr_lidx)
            # print(arr_ridx)
            # print(arr_dist)

            nfound += 1

            # print("last was", last, "r_e[h] is", r_e[h])
            if r_e[h] != last:
                nfound_local += 1

            last = r_e[h]
            h -= 1

        # print("while done. h:", h, "nfound_local:", nfound_local)

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_next_nonoverlapping64_all(const int64_t [::1] l_e, const int64_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int _dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int64) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int64_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    for i in range(len_l):

        h = r_idx[i]
        nfound_local = 0
        last = r_s[h]
        # print("-----" * 10)
        # print("i", i)

        while h < len_r and nfound_local < k[i]:

            # print("h", h, "nfound_local", nfound_local)

            if nfound == outarr_length - 1:
                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = r_s[h] - l_e[i]

            # print(arr_lidx)
            # print(arr_ridx)
            # print(arr_dist)

            nfound += 1

            # print("last was", last, "r_s[h] is", r_s[h])
            if r_s[h] != last:
                nfound_local += 1

            last = r_s[h]
            h += 1

        # print("while done. h:", h, "nfound_local:", nfound_local)
        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_previous_nonoverlapping64_k_distances(const int64_t [::1] l_s, const int64_t [::1] r_e,
                                                    const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_s)
    # cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int j = 0 # counter for r_e
    cdef int i = 0 # counter for l_s
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int last_dist = -1
    cdef int curr_dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int64) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int64_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    while i < len_l and j < len_r:

        h = r_idx[j]
        nfound_local = 0
        j += 1
        last_dist = -1

        while h >= 0 and nfound_local < k[i]:

            if nfound == outarr_length - 1:

                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = l_s[i] - r_e[h]
            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist

            nfound += 1

            if curr_dist != last_dist:
                nfound_local += 1

            curr_dist = last_dist

            last = r_e[h]
            h -= 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_next_nonoverlapping64_k_distances(const int64_t [::1] l_e, const int64_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int curr_dist = -1
    cdef int last_dist = -1


    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int64) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int64_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    while i < len_l and j < len_rx:

        h = r_idx[j]
        nfound_local = 0
        j += 1
        last = -1

        while h < len_r and nfound_local < k[i]:

            if nfound == outarr_length - 1:

                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = r_s[h] - l_e[i]

            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist

            nfound += 1

            if curr_dist != last_dist:
                nfound_local += 1

            last_dist = curr_dist

            last = r_s[h]
            h += 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_next_nonoverlapping64_first(const int64_t [::1] l_e, const int64_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    # print("nearest_next_nonoverlapping first")

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int curr_dist = -1
    cdef int last_dist = -1


    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int64) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int64_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    # print("ridx", list(r_idx))
    # print("lidx", list(l_idx))
    # print("l_e", list(l_e))
    # print("r_s", list(r_s))
    # print("i", i)
    # print("len_l", len_l)
    # print("len_rx", len_rx)
    # print("j", j)
    while i < len_l:

        nfound_local = 0
        last_dist = -1

        h = r_idx[i]

        while h < len_r and nfound_local < k[i]:

            # print("h", h)
            # print("rs", r_s[h])
            # print("le", l_e[i])

            if nfound >= outarr_length - 1:


                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = r_s[h] - l_e[i]

            if curr_dist != last_dist:
                # print("inserting ridx", h, "lidx", l_idx[i], "dist", curr_dist)
                ridx[nfound] = h
                lidx[nfound] = l_idx[i]
                dist[nfound] = curr_dist

                nfound += 1

                nfound_local += 1

                last_dist = curr_dist

            last = r_s[h]
            h += 1

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_previous_nonoverlapping64_first(const int64_t [::1] l_s, const int64_t [::1] r_e,
                                              const long [::1] l_idx, const long [::1] r_idx,
                                              const long [::1] k):

    # print("nearest previous nonoverlapping first")

    cdef int len_l = len(l_s)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int j = 0 # counter for r_e
    cdef int i = 0 # counter for l_s
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = len_l
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int last_dist = -1
    cdef int curr_dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int64) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef int64_t [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    # print("r_idx", list(r_idx))
    # print("r_e", list(r_e))
    # print("k", list(k))

    while i < len_l:

        last_dist = -1
        nfound_local = 0
        h = r_idx[i]

        while nfound_local < k[i] and h >= 0:  # and l_s[i] - r_e[h] >= 0:

            # print("h", h)
            # print("ls", l_s[i])
            # print("re", r_e[h])

            if nfound >= outarr_length - 1:
                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = l_s[i] - r_e[h]

            if curr_dist != last_dist:
                # print("inserting ridx", h, "lidx", l_idx[i], "dist", curr_dist)
                ridx[nfound] = h
                lidx[nfound] = l_idx[i]
                dist[nfound] = curr_dist
                last_dist = curr_dist

                nfound += 1

                nfound_local += 1

            last = r_e[h]
            h -= 1
            # print("h", h, "len_rx", len_rx)

        i += 1
        # print("i", i, "len_l", len_l)

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_next_nonoverlapping64_last(const int64_t [::1] l_e, const int64_t [::1] r_s,
                                        const long [::1] l_idx, const long [::1] r_idx, const long [::1] k):

    # print("  nearest_next_nonoverlapping last" * 5)

    cdef int len_l = len(l_e)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_s)

    cdef int j = 0
    cdef int i = 0
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = 0
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int curr_dist = -1
    cdef int last_dist = -1
    cdef int start_dist = -1


    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int64) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef long [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    for i in range(len_l):

        # print("i", i)
        nfound_local = 0
        h = r_idx[i]
        last_dist = r_s[h] - l_e[i]
        start_dist = last_dist

        while h < len_r and nfound_local < k[i]:

            # print("h", h)
            # print("rs", r_s[h])
            # print("le", l_e[i])

            if nfound >= outarr_length - 1:

                # print("--resizing--" * 5)
                # print("before resizing", arr_lidx)

                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                # print("after resizing", arr_lidx)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = r_s[h] - l_e[i]

            """Cases:
1. all dist per i equal. need to add last dist manually
2. dists unequal -> all will be added but last. Need to add last manually
"""
            # if unequal, begin searching for last for next distance
            if curr_dist != last_dist:
                # print("if unequal, begin searching for last for next distance")
                nfound += 1
                nfound_local += 1

            # for each entry, add distance
            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist


            # print("lidx", list(lidx))
            # print("ridx", list(ridx))
            # print("dist", list(dist))

            last_dist = curr_dist

            h += 1

        # in case of all equal, need to add manually
        if start_dist == last_dist and k[i]:
            # print("in case of all equal, need to add manually")
            ridx[nfound] = h - 1
            lidx[nfound] = l_idx[i]
            dist[nfound] = last_dist
            # print("lidx", list(lidx))
            # print("ridx", list(ridx))
            # print("dist", list(dist))
            nfound += 1


        # in case last dist was not added
        elif nfound_local < k[i] and dist[nfound - 1] != last_dist:
            # print("FINALLY inserting lidx", l_idx[i], "ridx", h - 1, "dist", curr_dist)

            ridx[nfound] = h - 1
            lidx[nfound] = l_idx[i]
            dist[nfound] = last_dist
            nfound += 1
            # print("lidx", list(lidx))
            # print("ridx", list(ridx))
            # print("dist", list(dist))

        i += 1

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef nearest_previous_nonoverlapping64_last(const int64_t [::1] l_s, const int64_t [::1] r_e,
                                              const long [::1] l_idx, const long [::1] r_idx,
                                              const long [::1] k):

    # print("  nearest previous nonoverlapping last" * 5)

    cdef int len_l = len(l_s)
    cdef int len_rx = len(r_idx)
    cdef int len_r = len(r_e)

    cdef int i = 0 # counter for r_e
    cdef int h = 0

    cdef int outarr_length = len_l
    cdef int outarr_length_old = -1
    cdef int nfound = 0
    cdef int nfound_local = 0
    cdef int last = -1
    cdef int last_dist = -1
    cdef int curr_dist = -1

    arr_lidx = np.ones(len_l, dtype=np.long) * -1
    arr_ridx = np.ones(len_l, dtype=np.long) * -1
    arr_dist = np.ones(len_l, dtype=np.int64) * -1

    cdef long [::1] lidx
    cdef long [::1] ridx
    cdef long [::1] dist

    lidx = arr_lidx
    ridx = arr_ridx
    dist = arr_dist

    # print("l_idx", list(l_idx))
    # print("r_idx", list(r_idx))
    # print("r_e", list(r_e))
    # print("k", list(k))
    # print("len_l", len_l)

    for i in range(len_l):

        nfound_local = 0
        h = r_idx[i]
        last_dist = -1

        # print("----" * 5)
        # print("i", i)

        while nfound_local < k[i] and h >= 0:  # and l_s[i] - r_e[h] >= 0:

            # print("***" * 3)
            # print("h", h)
            # print("ls", l_s[i])
            # print("re", r_e[h])

            if nfound >= outarr_length - 1:

                outarr_length_old = outarr_length
                outarr_length = (outarr_length * 2) + 1
                arr_lidx = np.resize(arr_lidx, outarr_length)
                arr_lidx[outarr_length_old:] = -1
                # print("after resizing", arr_lidx)
                lidx = arr_lidx
                arr_ridx = np.resize(arr_ridx, outarr_length)
                arr_ridx[outarr_length_old:] = -1
                ridx = arr_ridx
                arr_dist = np.resize(arr_dist, outarr_length)
                arr_dist[outarr_length_old:] = -1
                dist = arr_dist

            curr_dist = l_s[i] - r_e[h]

            # print("inserting lidx", l_idx[i], "ridx", h, "dist", curr_dist, "i", i)
            ridx[nfound] = h
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist

            # print("curr_dist", curr_dist, "vs last_dist", last_dist)
            if curr_dist != last_dist:
                nfound += 1
                nfound_local += 1
                # print("ridx", list(ridx))
                # print("nfound", nfound)
            last_dist = curr_dist

            # print("lidx", list(lidx))
            # print("ridx", list(ridx))
            # print("dist", list(dist))

            h -= 1

        if nfound > 0 and nfound_local < k[i] and dist[nfound - 1] != curr_dist:

            # print("FINALLY inserting lidx", l_idx[i], "ridx", h + 1, "dist", curr_dist)
            ridx[nfound] = h + 1
            lidx[nfound] = l_idx[i]
            dist[nfound] = curr_dist
            nfound += 1

            # print("lidx", list(lidx)[:nfound])
            # print("ridx", list(ridx)[:nfound])
            # print("dist", list(dist)[:nfound])

        i += 1
        # print("i", i, "len_l", len_l)

    return arr_lidx[:nfound], arr_ridx[:nfound], arr_dist[:nfound]
