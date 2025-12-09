
import numpy as np
from numba import njit, prange
from math import fabs, pow

@njit(parallel=False, cache=False)
def misfit_L2(data_data, greens_data, greens_greens,
                 sources, groups, weights,
                 hybrid_norm, dt, NPAD1, NPAD2,
                 debug_level, msg_start, msg_stop, msg_percent):
   
    NSRC = sources.shape[0]
    NSTA = weights.shape[0]
    NC = weights.shape[1]
    NG = sources.shape[1]
    NGRP = groups.shape[0]
    NPAD = NPAD1 + NPAD2 + 1

    results = np.zeros((NSRC, 1), dtype=np.float64)
    cc = np.zeros(NPAD, dtype=np.float64)

    if msg_percent > 0:
      msg_interval = msg_percent/100.*msg_stop
      msg_count = 100./msg_percent*msg_start/msg_stop
      iter = msg_start
      next_iter = msg_count*msg_interval
    else:
      iter = 0
      next_iter = NSRC+1


    for isrc in range(NSRC):
        # display progress message
        if iter >= next_iter:
            print("    percent finished: ", (msg_percent*msg_count))
            msg_count += 1;
            next_iter = msg_count*msg_interval
        iter += 1

        L2_sum = 0.0

        for ista in range(NSTA):
            for igrp in range(NGRP):
                cc[:] = 0.0

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

                    # s^2 term
                    for j1 in range(NG):
                        for j2 in range(NG):
                            L2_tmp += sources[isrc,j1] * sources[isrc,j2] *\
                                          greens_greens[ista, ic, itpad, j1, j2]

                    # d^2 term
                    L2_tmp += data_data[ista, ic]

                    # 2sd term
                    for ig in range(NG):
                        L2_tmp -= 2.0 * greens_data[ista, ic, ig, itpad] * sources[isrc, ig]

                    if hybrid_norm == 0:
                        L2_sum += dt * weights[ista, ic] * L2_tmp
                    else:
                        L2_sum += dt * weights[ista, ic] * pow(L2_tmp, 0.5)

        results[isrc] = L2_sum

    return results

