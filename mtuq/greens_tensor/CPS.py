
import obspy
import numpy as np

from mtuq.greens_tensor.base import GreensTensor as GreensTensorBase


class GreensTensor(GreensTensorBase):
    """
    CPS Green's tensor object

    Overloads base class with machinery for working with CPS-style
    Green's functions

    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)

        if 'type:greens' not in self.tags:
            self.tags += ['type:greens']

        if 'type:velocity' not in self.tags or 'type:displacement' not in self.tags:
            self.tags += ['type:velocity']

        if 'units:m' not in self.tags:
            self.tags += ['units:m']

    def _precompute(self):
        """ Computes NumPy arrays used by get_synthetics
        """
        if self.include_mt:
            self._precompute_mt()

        if self.include_force:
            self._precompute_force()


    def _precompute_mt(self):
        """ Recombines CPS time series so they can be used in straightforward
        linear combination with Mrr,Mtt,Mpp,Mrt,Mrp,Mtp
        """

        array = self._array
        phi = np.deg2rad(self.azimuth)
        _j = 0

        # The formulas below were obtained by reverse engineering FK

        for _i, component in enumerate(self.components):
            if component=='Z':
                ZSS = self.select(channel="ZSS")[0].data
                ZDS = self.select(channel="ZDS")[0].data
                ZDD = self.select(channel="ZDD")[0].data
                ZEP = self.select(channel="ZEP")[0].data
                array[_i, _j+0, :] =  ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                array[_i, _j+1, :] = -ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                array[_i, _j+2, :] =  ZDD/3. + ZEP/3.
                array[_i, _j+3, :] =  ZSS * np.sin(2*phi)
                array[_i, _j+4, :] =  ZDS * np.cos(phi)
                array[_i, _j+5, :] =  ZDS * np.sin(phi)

            elif component=='R':
                RSS = self.select(channel="RSS")[0].data
                RDS = self.select(channel="RDS")[0].data
                RDD = self.select(channel="RDD")[0].data
                REP = self.select(channel="REP")[0].data
                array[_i, _j+0, :] =  RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                array[_i, _j+1, :] = -RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                array[_i, _j+2, :] =  RDD/3. + REP/3.
                array[_i, _j+3, :] =  RSS * np.sin(2*phi)
                array[_i, _j+4, :] =  RDS * np.cos(phi)
                array[_i, _j+5, :] =  RDS * np.sin(phi)

            elif component=='T':
                TSS = self.select(channel="TSS")[0].data
                TDS = self.select(channel="TDS")[0].data
                array[_i, _j+0, :] =  TSS/2. * np.sin(2*phi)
                array[_i, _j+1, :] = -TSS/2. * np.sin(2*phi)
                array[_i, _j+2, :] =  0.
                array[_i, _j+3, :] = -TSS * np.cos(2*phi)
                array[_i, _j+4, :] =  TDS * np.sin(phi)
                array[_i, _j+5, :] = -TDS * np.cos(phi)

            else:
                raise ValueError

        #
        # CPS uses a north-east-down basis convention, while mtuq uses an
        # up-south-east basis convention, so a permutation is necessary
        #
        array_copy = array.copy()
        array[:, 0, :] =  array_copy[:, 2, :]
        array[:, 1, :] =  array_copy[:, 0, :]
        array[:, 2, :] =  array_copy[:, 1, :]
        array[:, 3, :] =  array_copy[:, 4, :]
        array[:, 4, :] = -array_copy[:, 5, :]
        array[:, 5, :] = -array_copy[:, 3, :]


    def _precompute_force(self):
        raise NotImplementedError()



