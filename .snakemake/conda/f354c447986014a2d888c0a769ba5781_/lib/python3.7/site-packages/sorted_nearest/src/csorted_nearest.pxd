# cdef extern from "sorted_nearest/src/utarray.h":

#     ctypedef struct UT_icd:
#         pass

#     ctypedef struct UT_array:
#         pass

#     const UT_icd ut_int_icd

#     void utarray_new(UT_array *a, UT_icd *icd)
#     int utarray_len(UT_array *a)
#     int* utarray_eltptr(UT_array *a, int j)
#     void utarray_push_back(UT_array *a, void *p)
#     void utarray_free(UT_array *a)

# cdef extern from "limits.h":
#     int INT_MAX
