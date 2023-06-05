from libc.stdint cimport int32_t, int64_t

cimport cython

import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef get_all_ties(const int64_t[::1] lx, const int64_t[::1] ids, const int64_t[::1] dist, int k):

    """all fetches until you have k nearest intervals and then all intervals with same distance."""

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

    while i < length:

        # print("------" * 10)
        # print("i", i)
        # print("ids[i]", ids[i])
        # print("last_id", last_id)
        # print("current_found", current_found)
        # print("nfound", nfound)
        # print("length", length)
        first_i = i

        if ids[i] != last_id:
            # we have come to a new id, need to start counting again
            # print("** if **")
            # print("i", i, "length", length, "ids[i]", ids[i], "last_id", last_id, "dist[i]", last_dist, "current_found", current_found)

            current_found = 0
            current_id = ids[i]
            last_id = ids[i]
            # continue

        # while have not found k different ties, continue
        while i < length and ids[i] == last_id and current_found < k:
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
cpdef get_different_ties(const int64_t[::1] lx, const int64_t[::1] ids, const int64_t[::1] dist, int k):

    """fetch all ties until you have all intervals with k distances"""

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
        while i < length and ids[i] == last_id and current_found < k:
            # print("**first while**")
            # print("i", i, "length", length, "ids[i]", ids[i], "last_id", last_id, "dist[i]", last_dist, "current_found", current_found)
            lidx[nfound] = lx[i]
            # print("lidx", arr_lidx)
            last_dist = dist[i]
            nfound += 1
            i += 1

            if i < length and dist[i] != last_dist:
                current_found += 1

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
