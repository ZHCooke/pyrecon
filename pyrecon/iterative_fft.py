"""Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591) algorithm."""

from .recon import BaseReconstruction, format_positions_wrapper, format_positions_weights_wrapper
from . import utils
import numpy as np

class IterativeFFTReconstruction(BaseReconstruction):
    """
    Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591)
    field-level (as opposed to :class:`IterativeFFTParticleReconstruction`) algorithm.
    """
    _compressed = True
    _f_z = True
    _bias_z = True

    @format_positions_weights_wrapper
    def assign_data(self, positions, weights=None, **kwargs):
        """
        Assign (paint) data to :attr:`mesh_data` and store positions for later use.
        """
        if weights is None:
            weights = np.ones_like(positions, shape=(len(positions),))

        if getattr(self, 'mesh_data', None) is None:
            self.mesh_data = self.pm.create(type='real', value=0.)
            self._positions_data = positions  # Store positions here!
            self._weights_data = weights
        else:
            self._positions_data = np.concatenate([self._positions_data, positions], axis=0)
            self._weights_data = np.concatenate([self._weights_data, weights], axis=0)

        self._paint(positions, weights=weights, out=self.mesh_data)


    def run(self, niterations=3):
        """
        Run reconstruction, i.e. compute Zeldovich displacement fields :attr:`mesh_psi`.

        Parameters
        ----------
        niterations : int, default=3
            Number of iterations.
        """
        self._iter = 0
        self.mesh_delta_real = self.mesh_delta.copy()
        for iter in range(niterations):
            self._iterate()
        del self.mesh_delta
        self.mesh_psi = self._compute_psi()
        del self.mesh_delta_real

    def _iterate(self):
        if self.mpicomm.rank == 0:
            self.log_info('Running iteration {:d}.'.format(self._iter))
        # This is an implementation of eq. 22 and 24 in https://arxiv.org/pdf/1504.02591.pdf
        # \delta_{g,\mathrm{real},n} is self.mesh_delta_real
        # \delta_{g,\mathrm{red}} is self.mesh_delta
        # First compute \delta(k)/k^{2} based on current \delta_{g,\mathrm{real},n} to estimate \phi_{\mathrm{est},n} (eq. 24)
        delta_k = self.mesh_delta_real.r2c()
        for kslab, slab in zip(delta_k.slabs.x, delta_k.slabs):
            utils.safe_divide(slab, sum(kk**2 for kk in kslab), inplace=True)

        self.mesh_delta_real = self.mesh_delta.copy()
        # Now compute \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r}
        # In the plane-parallel case (self.los is a given vector), this is simply \beta IFFT((\hat{k} \cdot \hat{\eta})^{2} \delta(k))
        if self.los is not None:
            # global los
            disp_deriv_k = delta_k.copy()
            for kslab, slab in zip(disp_deriv_k.slabs.x, disp_deriv_k.slabs):
                slab[...] *= sum(kk * ll for kk, ll in zip(kslab, self.los))**2  # delta_k already divided by k^{2}
            factor = self.beta
            # remove RSD part
            if self._iter == 0:
                # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                factor /= (1. + self.beta)
            self.mesh_delta_real -= factor * disp_deriv_k.c2r()
            del disp_deriv_k
        else:
            # In the local los case, \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r} is:
            # \beta \partial_{i} \partial_{j} \phi_{\mathrm{est},n} \hat{r}_{j} \hat{r}_{i}
            # i.e. \beta IFFT(k_{i} k_{j} \delta(k) / k^{2}) \hat{r}_{i} \hat{r}_{j} => 6 FFTs
            for iaxis in range(delta_k.ndim):
                for jaxis in range(iaxis, delta_k.ndim):
                    disp_deriv = delta_k.copy()
                    for kslab, islab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs.i, disp_deriv.slabs):
                        mask = (islab[iaxis] != self.nmesh[iaxis] // 2) & (islab[jaxis] != self.nmesh[jaxis] // 2)
                        mask |= (islab[iaxis] == self.nmesh[iaxis] // 2) & (islab[jaxis] == self.nmesh[jaxis] // 2)
                        slab[...] *= kslab[iaxis] * kslab[jaxis] * mask  # delta_k already divided by k^{2}
                    disp_deriv = disp_deriv.c2r()
                    for rslab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs):
                        rslab = self._transform_rslab(rslab)
                        slab[...] *= utils.safe_divide(rslab[iaxis] * rslab[jaxis], sum(rr**2 for rr in rslab))
                    factor = (1. + (iaxis != jaxis)) * self.beta  # we have j >= i and double-count j > i to account for j < i
                    if self._iter == 0:
                        # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                        factor /= (1. + self.beta)
                    # remove RSD part
                    self.mesh_delta_real -= factor * disp_deriv
        self._iter += 1

    def _compute_psi(self):
        # Compute Zeldovich displacements given reconstructed real space density
        delta_k = self.mesh_delta_real.r2c()
        psis = []
        for iaxis in range(delta_k.ndim):
            psi = delta_k.copy()
            for kslab, islab, slab in zip(psi.slabs.x, psi.slabs.i, psi.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2
                slab[...] *= 1j * utils.safe_divide(kslab[iaxis], sum(kk**2 for kk in kslab)) * mask
            psis.append(psi.c2r())
            del psi
        return psis
    
class HybridIterativeFFTReconstruction(IterativeFFTReconstruction):
    """
    Variation of the field-level iterative FFT reconstruction algorithm of 
    Burden et al. 2015 (https://arxiv.org/abs/1504.02591) implemented in 
    :class:`IterativeFFTReconstruction` that also shifts the galaxy positions 
    to remove RSD in each iterative step
    """

    def run(self, niterations=3):
        """
        Run reconstruction, i.e. compute Zeldovich displacement fields :attr:`mesh_psi` and also 
        the reconstructed data real-space positions :attr:`_positions_rec_data`.

        Parameters
        ----------
        niterations : int, default=3
            Number of iterations.
        """
        self._iter = 0
        self.mesh_delta_real = self.mesh_delta.copy()
        self._positions_rec_data = self._positions_data.copy()
        for iter in range(niterations):
            self.alt_mesh_psi = self._iterate(return_psi=iter == niterations - 1)
        del self.mesh_delta
        self.mesh_psi = self._compute_psi()
        del self.mesh_delta_real
        # wrap recon positions (https://github.com/cosmodesi/pyrecon/blob/3508ab0a1109cce686bbc74bc1718c7787f69f52/
        # pyrecon/iterative_fft_particle.py#L220C15-L220C34 suggests _positions_rec_data already wrapped during
        # iterations, but I can't see where, so do this again for safety)
        if self.wrap: self._positions_rec_data = self._wrap(self._positions_rec_data)

    def _iterate(self, return_psi=False):
        if self.mpicomm.rank == 0:
            self.log_info('Running iteration {:d}.'.format(self._iter))
            
        # This routine is an implementation of the iterative procedure defined in eqs. 20, 22 and 24 
        # of https://arxiv.org/pdf/1504.02591.pdf. In each iteration:
        # self.mesh_delta contains \delta_{g,\mathrm{red}}, the original redshift-space density
        # self.mesh_delta_real contains \delta_{g,\mathrm{real},n}, the iterative estimate of the real-space density
        # (in the first iteration this is equal to the redshift-space density)
        # The estimate of \phi_{\mathrm{est},n+1} is obtained from the current \delta_{g,\mathrm{real},n} (eq. 24)
        
        # First compute \delta(k)/k^{2} based on current \delta_{g,\mathrm{real},n}
        delta_k = self.mesh_delta_real.r2c()
        for kslab, slab in zip(delta_k.slabs.x, delta_k.slabs):
            utils.safe_divide(slab, sum(kk**2 for kk in kslab), inplace=True)

        # Before computing the updated density field, start by estimating the shift to the data positions
        # (but we don't use the shifted positions to estimate the new field)
        if self.mpicomm.rank == 0:
            self.log_info('Computing data shifts.')
        shifts = np.empty_like(self._positions_rec_data)
        psis = []
        for iaxis in range(delta_k.ndim):
            # No need to compute psi on axis where los is 0
            if not return_psi and self.los is not None and self.los[iaxis] == 0:
                shifts[:, iaxis] = 0.
                continue

            psi = delta_k.copy()
            for kslab, islab, slab in zip(psi.slabs.x, psi.slabs.i, psi.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2
                slab[...] *= 1j * kslab[iaxis] * mask

            psi = psi.c2r()
            # Reading shifts at reconstructed data real-space positions
            shifts[:, iaxis] = self._readout(psi, self._positions_rec_data)
            if return_psi: psis.append(psi)
            del psi
        if self.los is None:
            los = utils.safe_divide(self._positions_data, utils.distance(self._positions_data)[:, None])
        else:
            los = self.los
        # For the first loop we add extra shift to approximately remove RSD component from psi to speed up convergence
        # See Burden et al. 2015: 1504.02591v2, eq. 12 (flat sky approximation)
        if self._iter == 0:
            shifts -= self.beta / (1 + self.beta) * np.sum(shifts * los, axis=-1)[:, None] * los
        # Now estimate the real-space positions, which are stored in self._positions_rec_data (but not used for anything else)
        self._positions_rec_data = self._positions_data - self.f * np.sum(shifts * los, axis=-1)[:, None] * los

        # Now compute the next estimate of the real-space field \delta_{g,\mathrm{real},n+1}
        self.mesh_delta_real = self.mesh_delta.copy()  # first part is based on the redshift-space density  
        # Now compute second term \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r} (eq 22)
        # In the plane-parallel case (self.los is a given vector), this is simply \beta IFFT((\hat{k} \cdot \hat{\eta})^{2} \delta(k))
        if self.los is not None:
            # global los
            disp_deriv_k = delta_k.copy()
            for kslab, slab in zip(disp_deriv_k.slabs.x, disp_deriv_k.slabs):
                slab[...] *= sum(kk * ll for kk, ll in zip(kslab, self.los))**2  # delta_k already divided by k^{2}
            factor = self.beta
            # remove RSD part
            if self._iter == 0:
                # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                factor /= (1. + self.beta)
            self.mesh_delta_real -= factor * disp_deriv_k.c2r()
            del disp_deriv_k
        else:
            # In the local los case, \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r} is:
            # \beta \partial_{i} \partial_{j} \phi_{\mathrm{est},n} \hat{r}_{j} \hat{r}_{i}
            # i.e. \beta IFFT(k_{i} k_{j} \delta(k) / k^{2}) \hat{r}_{i} \hat{r}_{j} => 6 FFTs
            for iaxis in range(delta_k.ndim):
                for jaxis in range(iaxis, delta_k.ndim):
                    disp_deriv = delta_k.copy()
                    for kslab, islab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs.i, disp_deriv.slabs):
                        mask = (islab[iaxis] != self.nmesh[iaxis] // 2) & (islab[jaxis] != self.nmesh[jaxis] // 2)
                        mask |= (islab[iaxis] == self.nmesh[iaxis] // 2) & (islab[jaxis] == self.nmesh[jaxis] // 2)
                        slab[...] *= kslab[iaxis] * kslab[jaxis] * mask  # delta_k already divided by k^{2}
                    disp_deriv = disp_deriv.c2r()
                    for rslab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs):
                        rslab = self._transform_rslab(rslab)
                        slab[...] *= utils.safe_divide(rslab[iaxis] * rslab[jaxis], sum(rr**2 for rr in rslab))
                    factor = (1. + (iaxis != jaxis)) * self.beta  # we have j >= i and double-count j > i to account for j < i
                    if self._iter == 0:
                        # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                        factor /= (1. + self.beta)
                    # remove RSD part
                    self.mesh_delta_real -= factor * disp_deriv
        self._iter += 1
        if return_psi:
            return psis

    def _compute_psi(self):
        # Compute Zeldovich displacements given reconstructed real space density
        delta_k = self.mesh_delta_real.r2c()
        psis = []
        for iaxis in range(delta_k.ndim):
            psi = delta_k.copy()
            for kslab, islab, slab in zip(psi.slabs.x, psi.slabs.i, psi.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2
                slab[...] *= 1j * utils.safe_divide(kslab[iaxis], sum(kk**2 for kk in kslab)) * mask
            psis.append(psi.c2r())
            del psi
        return psis

    @format_positions_wrapper(return_input_type=False)
    def read_shifts(self, positions, field='disp+rsd'):
        """
        Read displacement at input positions, following same routine in :class:`IterativeFFTParticleReconstruction`.

        Note
        ----
        Data shifts are read at the reconstructed real-space positions,
        while random shifts are read at the redshift-space positions.

        Parameters
        ----------
        positions : array of shape (N, 3), string
            Cartesian positions.
            Pass string 'data' to get the displacements for the input data positions passed to :meth:`assign_data`.
            Note that in this case, shifts are read at the reconstructed data real-space positions.

        field : string, default='disp+rsd'
            Either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        Returns
        -------
        shifts : array of shape (N, 3)
            Displacements.
        """
        field = field.lower()
        allowed_fields = ['disp', 'rsd', 'disp+rsd']
        if field not in allowed_fields:
            raise ReconstructionError('Unknown field {}. Choices are {}'.format(field, allowed_fields))

        def _read_shifts(positions):
            shifts = np.empty_like(positions)
            for iaxis, psi in enumerate(self.mesh_psi):
                shifts[:, iaxis] = self._readout(psi, positions)
            return shifts

        # the default way of doing things with data: use the reconstructed real-space 
        # positions computed during iterations and add displacement shifts on top
        if isinstance(positions, str) and positions == 'data':
            shifts = _read_shifts(self._positions_rec_data)
            if field == 'disp':
                return shifts
            rsd = self._positions_data - self._positions_rec_data
            if field == 'rsd':
                return rsd
            # field == 'disp+rsd'
            shifts += rsd
            return shifts

        # also have the option to compute displacement and rsd shifts at specified input
        # positions: used for randoms or as a non-standard option for data
        if self.wrap: positions = self._wrap(positions)  # wrap here for local los
        shifts = _read_shifts(positions)  # aleady wrapped
        if field == 'disp':
            return shifts
        # have to compute rsd shifts from scratch in this scenario
        if self.los is None:
            los = utils.safe_divide(positions, utils.distance(positions)[:, None])
        else:
            los = self.los.astype(positions.dtype)
        if self.f_callable is None:
            f = self.f
        else:
            f = self.f_callable(utils.distance(positions))[..., None]
        rsd = f * (np.sum(shifts * los, axis=-1)[:, None] * los)
        if field == 'rsd':
            return rsd

        # for field == 'disp+rsd', here we follow the 'usual' prescription to match IterativeFFTReconstruction
        shifts += rsd
        return shifts

        # but alternative IterativeFFTParticleReconstruction approach would be to remove RSD 
        # first then remove the Zeldovich displacement, as below
        # real_positions = positions - rsd
        # diff = real_positions - self.offset
        # if (not self.wrap) and any(self.mpicomm.allgather(np.any((diff < 0) | (diff > self.boxsize - self.cellsize)))):
        #     if self.mpicomm.rank == 0:
        #         self.log_warning('Some particles are out-of-bounds.')
        # shifts = _read_shifts(real_positions)

        # return shifts + rsd

    @format_positions_wrapper(return_input_type=True)
    def read_shifted_positions(self, positions, field='disp+rsd'):
        """
        Read final shifted positions i.e. the difference ``positions - self.read_shifts(positions, field=field)``.
        Output (and input) positions are wrapped if :attr:`wrap`.

        Parameters
        ----------
        positions : array of shape (N, 3), string
            Cartesian positions.
            Pass string 'data' to get the shift positions for the input data positions passed to :meth:`assign_data`.
            Note that in this case, shifts are read at the reconstructed data real-space positions.

        field : string, default='disp+rsd'
            Apply either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        Returns
        -------
        positions : array of shape (N, 3)
            Shifted positions.
        """
        shifts = self.read_shifts(positions, field=field, position_type='pos', mpiroot=None)
        if isinstance(positions, str) and positions == 'data':
            positions = self._positions_data
        positions = positions - shifts
        if self.wrap: positions = self._wrap(positions)
        return positions




