
import numpy as np

from mtuq.grid import FullMomentTensorGridRandom
from mtuq.util.math import to_mij, from_mij
from matplotlib import pyplot


if __name__=='__main__':

    grid = FullMomentTensorGridRandom(
        npts=100000,
        magnitudes=[3.])

    grid.callback = None

    parameters = ('rho','v','w','kappa','sigma','h')

    verbose = 1


    # array to generate error histograms
    mt_array = grid.to_array()
    error_array = np.zeros((len(grid), 6))

    for _i,mt in enumerate(grid):
        print(_i)

        mtp = from_mij(to_mij(*mt))
        error = mt - mtp

        for _j, key in enumerate(parameters):
            if verbose > 0:
                print(f'  {key}')
                print(f'  {mt[_j]}')
                print(f'  {mtp[_j]}')
                print(f'  error: {error[_j]}\n')

            #mt_array[_i,_j] = mt[_j]
            error_array[_i,_j] = error[_j]

        print()


    fig, axes = pyplot.subplots(2,3,figsize=(16,12))

    for _j, axis in enumerate(axes.ravel()):
        axis.hist(error_array[:,_j])
        axis.set_title(parameters[_j])

    pyplot.savefig('error.png')


    fig, axes = pyplot.subplots(2,3,figsize=(16,12))

    for _j, axis in enumerate(axes.ravel()):
        axis.hist(error_array[:,_j]/mt_array[:,_j])
        axis.set_title(parameters[_j])

    pyplot.savefig('normalized_error.png')


