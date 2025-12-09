# misfit.pyx
import numpy as np
cimport numpy as np
from libc.math cimport fabs, pow

import cython
@cython.boundscheck(False)
@cython.wraparound(False)

def misfit(np.ndarray[np.float64_t, ndim=2] data_data,
           np.ndarray[np.float64_t, ndim=4] greens_data,
           np.ndarray[np.float64_t, ndim=5] greens_greens,
           np.ndarray[np.float64_t, ndim=2] sources,
           np.ndarray[np.float64_t, ndim=2] groups,
           np.ndarray[np.float64_t, ndim=2] weights,
           int hybrid_norm,
           double dt,
           int NPAD1, int NPAD2,
           int debug_level,
           int msg_start, int msg_stop, int msg_percent):

    cdef int NSRC = sources.shape[0]
    cdef int NSTA = weights.shape[0]
    cdef int NC = weights.shape[1]
    cdef int NG = sources.shape[1]
    cdef int NGRP = groups.shape[0]
    cdef int NPAD = NPAD1 + NPAD2 + 1

    cdef np.ndarray[np.float64_t, ndim=1] cc = np.zeros(NPAD, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] results = np.zeros((NSRC, 1), dtype=np.float64)

    cdef int isrc, ista, ic, ig, igrp, it, j1, j2, itpad, cc_argmax
    cdef double cc_max, L2_sum, L2_tmp, source_val

    for isrc in range(NSRC):
        cc[:] = 0
        L2_sum = 0.0

        for ista in range(NSTA):
            for igrp in range(NGRP):
                cc[:] = 0

                for ic in range(NC):
                    if groups[igrp, ic] == 0:
                        continue
                    if fabs(weights[ista, ic]) < 1e-6:
                        continue

                    for ig in range(NG):
                        source_val = sources[isrc, ig]
                        for it in range(NPAD):
                            cc[it] += greens_data[ista, ic, ig, it] * source_val

                cc_max = -np.inf
                cc_argmax = 0
                for it in range(NPAD):
                    if cc[it] > cc_max:
                        cc_max = cc[it]
                        cc_argmax = it
                itpad = cc_argmax

                for ic in range(NC):
                    if groups[igrp, ic] == 0:
                        continue
                    if fabs(weights[ista, ic]) < 1e-6:
                        continue

                    L2_tmp = 0.0
                    for j1 in range(NG):
                        for j2 in range(NG):
                            L2_tmp += sources[isrc,j1] * sources[isrc, j2] *\
                                          greens_greens[ista, ic, itpad, j1, j2]

                    L2_tmp += data_data[ista, ic]
                    for ig in range(NG):
                        L2_tmp -= 2.0 * greens_data[ista, ic, ig, itpad] * sources[isrc, ig]

                    if hybrid_norm == 0:
                        L2_sum += dt * weights[ista, ic] * L2_tmp
                    else:
                        L2_sum += dt * weights[ista, ic] * pow(L2_tmp, 0.5)

        results[isrc,0] = L2_sum

    return results

