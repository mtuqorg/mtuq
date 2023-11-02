
import obspy
import numpy as np

from mtuq.greens_tensor.base import GreensTensor as GreensTensorBase



class GreensTensor(GreensTensorBase):
    """
    AxiSEM Green's tensor object

    Overloads base class with machinery for working with AxiSEM-style
    Green's functions

    .. note:

      AxiSEM Green's functions describe the impulse response of a radially-
      symmetric medium.  Time series represent vertical, radial, and transverse
      displacement in units of m*(N-m)^-1

      For the vertical and raidal components, there are four associated time
      series. For the tranverse component, there are two associated time
      series. Thus there are ten independent Green's tensor elements altogether,
      which is fewer than in the case of a general inhomogeneous medium

    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)

        if 'type:greens' not in self.tags:
            self.tags += ['type:greens']

        if 'type:displacement' not in self.tags:
            self.tags += ['type:displacement']

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
        """ Recombines AxiSEM time series so they can be used in straightforward
        linear combination with Mrr,Mtt,Mpp,Mrt,Mrp,Mtp
        """
        array = self._array
        phi = np.deg2rad(self.azimuth)
        _j = 0

        # The formulas below are copied directly from Minson2008

        for _i, component in enumerate(self.components):
            if component=='Z':
                ZSS = +self.select(channel="ZSS")[0].data
                ZDS = -self.select(channel="ZDS")[0].data
                ZDD = +self.select(channel="ZDD")[0].data
                ZEP = +self.select(channel="ZEP")[0].data
                array[_i, _j+0, :] =  ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                array[_i, _j+1, :] = -ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEP/3.
                array[_i, _j+2, :] =  ZDD/3. + ZEP/3.
                array[_i, _j+3, :] =  ZSS * np.sin(2*phi)
                array[_i, _j+4, :] =  ZDS * np.cos(phi)
                array[_i, _j+5, :] =  ZDS * np.sin(phi)

            elif component=='R':
                RSS = +self.select(channel="RSS")[0].data
                RDS = -self.select(channel="RDS")[0].data
                RDD = +self.select(channel="RDD")[0].data
                REP = +self.select(channel="REP")[0].data
                array[_i, _j+0, :] =  RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                array[_i, _j+1, :] = -RSS/2. * np.cos(2*phi) - RDD/6. + REP/3.
                array[_i, _j+2, :] =  RDD/3. + REP/3.
                array[_i, _j+3, :] =  RSS * np.sin(2*phi)
                array[_i, _j+4, :] =  RDS * np.cos(phi)
                array[_i, _j+5, :] =  RDS * np.sin(phi)

            elif component=='T':
                TSS = -self.select(channel="TSS")[0].data
                TDS = +self.select(channel="TDS")[0].data
                array[_i, _j+0, :] = TSS/2. * np.sin(2*phi)
                array[_i, _j+1, :] = -TSS/2. * np.sin(2*phi)
                array[_i, _j+2, :] = 0.
                array[_i, _j+3, :] = -TSS * np.cos(2*phi)
                array[_i, _j+4, :] = TDS * np.sin(phi)
                array[_i, _j+5, :] = -TDS * np.cos(phi)

        #
        # Minson2008 uses a north-east-down basis convention, while mtuq uses an
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
        """ Computes NumPy arrays used in force linear combination
        """
        array = self._array
        phi = np.deg2rad(self.azimuth)

        _j = 0
        if self.include_mt:
            _j += 6

        for _i, component in enumerate(self.components):
            if component=='Z':
                Z0 = self.select(channel="Z0")[0].data
                Z1 = self.select(channel="Z1")[0].data
                Z2 = self.select(channel="Z2")[0].data
                array[_i, _j+0, :] = Z0
                array[_i, _j+1, :] = Z1
                array[_i, _j+2, :] = Z2

            elif component=='R':
                R0 = self.select(channel="R0")[0].data
                R1 = self.select(channel="R1")[0].data
                R2 = self.select(channel="R2")[0].data

                #
                # A sign change is necessary for the radial component 
                # returned by Instaseis/syngine:
                #

                R0 = -R0
                R1 = -R1
                R2 = -R2

                #
                # - For more information see:
                #   https://github.com/krischer/instaseis/issues/77
                #   https://github.com/krischer/instaseis/issues/82
                #
                # - It is important that a copy of the numeric trace data
                #   is created (so that the fix remains correct even if
                #   _precompute() is called multiple times)
                #
                # - See also test_syngine.py, which tests for changes to
                #   Instaseis/syngine itself, which may obviate the above
                #   workaround
                #

                array[_i, _j+0, :] = R0
                array[_i, _j+1, :] = R1
                array[_i, _j+2, :] = R2

            elif component=='T':
                T0 = self.select(channel="T0")[0].data
                T1 = self.select(channel="T1")[0].data
                T2 = self.select(channel="T2")[0].data
                array[_i, _j+0, :] = T0
                array[_i, _j+1, :] = T1
                array[_i, _j+2, :] = T2



