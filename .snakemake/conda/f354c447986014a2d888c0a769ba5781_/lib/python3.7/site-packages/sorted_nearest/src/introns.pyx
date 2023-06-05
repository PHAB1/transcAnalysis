
from libc.stdint cimport int32_t

cimport cython

import numpy as np

def find_introns(gene_starts, gene_ends, gene_ids,
                 exon_starts, exon_ends, exon_gene_ids):

    if gene_starts.dtype == np.long:
        return find_introns64(gene_starts, gene_ends, gene_ids, exon_starts, exon_ends, exon_gene_ids)
    elif gene_starts.dtype == np.int32:
        return find_introns32(gene_starts, gene_ends, gene_ids, exon_starts, exon_ends, exon_gene_ids)
    else:
        raise Exception("Starts/Ends not int64 or int32: " + str(gene_starts.dtype))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef find_introns64(const long [::1] gene_starts, const long [::1] gene_ends, const long [::1] gene_ids,
                     const long [::1] exon_starts, const long [::1] exon_ends, const long [::1] x_gene_ids):

    cdef:
        exon_length = len(exon_starts)
        length = 2 * exon_length
        int n = 0
        int i = 0
        int j = 0
        long gene_start, gene_end, gene_id
        long [::1] output_start
        long [::1] output_end
        long [::1] output_id

    output_arr_start = np.ones(length, dtype=np.long) * -1
    output_arr_end = np.zeros(length, dtype=np.long) * -1
    output_arr_id = np.zeros(length, dtype=np.long) * -1

    output_id = output_arr_id
    output_start = output_arr_start
    output_end = output_arr_end

    for i in range(len(gene_starts)):

        gene_id = gene_ids[i]
        gene_start = gene_starts[i]
        gene_end = gene_ends[i]

        if gene_start != exon_starts[j]:
            output_start[n] = gene_start
            output_end[n] = exon_starts[j]
            output_id[n] = gene_id
            n += 1

        j += 1

        while j < exon_length and gene_id == x_gene_ids[j]:

            output_start[n] = exon_ends[j - 1]
            output_end[n] = exon_starts[j]
            output_id[n] = gene_id
            n += 1
            j += 1

        if exon_ends[j - 1] != gene_end:

            output_start[n] = exon_ends[j - 1]
            output_end[n] = gene_end
            output_id[n] = gene_id
            n += 1

    return output_arr_start[:n], output_arr_end[:n], output_arr_id[:n]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef find_introns32(const int32_t [::1] gene_starts, const int32_t [::1] gene_ends, const long [::1] gene_ids,
                     const int32_t [::1] exon_starts, const int32_t [::1] exon_ends, const long [::1] x_gene_ids):

    cdef:
        exon_length = len(exon_starts)
        length = 2 * exon_length
        int n = 0
        int i = 0
        int j = 0
        int gene_start, gene_end, gene_id
        int32_t [::1] output_start
        int32_t [::1] output_end
        int32_t [::1] output_id

    output_arr_start = np.ones(length, dtype=np.int32) * -1
    output_arr_end = np.zeros(length, dtype=np.int32) * -1
    output_arr_id = np.zeros(length, dtype=np.int32) * -1

    output_id = output_arr_id
    output_start = output_arr_start
    output_end = output_arr_end

    for i in range(len(gene_starts)):

        gene_id = gene_ids[i]
        gene_start = gene_starts[i]
        gene_end = gene_ends[i]
        # print("-----" * 5)
        # print("gene_id", gene_id)
        # print("gene_start", gene_start)
        # print("gene_end", gene_end)
        # print("j", j)
        # print()
        # assert gene_ids[i] == x_gene_ids[j], "gene_exon_not_equal"

        # print("gene_start != exon_starts[j]", gene_start != exon_starts[j])

        # assert gene_start <= exon_starts[j], "gene_start not smaller or equal to exon_end: {}, {}".format(gene_start, exon_starts[j])

        if gene_start != exon_starts[j]:
            # print("exon_start != gene_start")
            # print("insert start", gene_start)
            # print("insert end", exon_starts[j])
            # print()
            output_start[n] = gene_start
            output_end[n] = exon_starts[j]
            output_id[n] = gene_id
            n += 1

        j += 1

        # print("gene_start != exon_starts[j]", gene_start != exon_starts[j])
        # print("j", j)
        # if j < exon_length:
        #     print("gene_id == x_gene_ids[j]", gene_id == x_gene_ids[j], gene_id, x_gene_ids[j])
        while j < exon_length and gene_id == x_gene_ids[j]:
            # if exon_ends[j - 1] == 1217804:
            #     print(gene_id)
            #     raise
            # print("insert start", exon_ends[j - 1])
            # print("insert end", exon_starts[j])
            # print()

            output_start[n] = exon_ends[j - 1]
            output_end[n] = exon_starts[j]
            output_id[n] = gene_id
            n += 1
            j += 1

        # if j < exon_length:
            # print("while ended here:", gene_id, x_gene_ids[j])

        if exon_ends[j - 1] != gene_end:
            # print("exon_ends[j - 1] != gene_end", exon_ends[j - 1] != gene_end)
            # print("insert start", exon_ends[j - 1])
            # print("insert end", gene_end)
            # print()

            output_start[n] = exon_ends[j - 1]
            output_end[n] = gene_end
            output_id[n] = gene_id
            n += 1

    return output_arr_start[:n], output_arr_end[:n], output_arr_id[:n]
