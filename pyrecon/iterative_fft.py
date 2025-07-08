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
    
class HybridIFFTReconstruction(IterativeFFTReconstruction):
    """
    Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591)
    field-level (as opposed to :class:`IterativeFFTParticleReconstruction`) algorithm.
    """
    _compressed = True
    _f_z = True
    _bias_z = True

    def run(self, niterations=3):
        """
        Run hybrid IFFT reconstruction, updating particle positions each iteration,
        storing every iteration’s ψ in a list, and returning it.

        Parameters
        ----------
        niterations : int, default=3
            Number of iterations to perform.

        Returns
        -------
        iter_psis : list of list of 3D arrays
            iter_psis[i] is the [ψ_x, ψ_y, ψ_z] fields from iteration i.
        """
        # reset and initialize
        self._iter               = 0
        self.mesh_delta_real     = self.mesh_delta.copy()
        self._positions_rec_data = self._positions_data.copy()

        # collect ψ from each iteration
        iter_psis = []

        for i in range(niterations):
            # run one iteration (must return psis)
            psis = self._iterate()
            iter_psis.append(psis)

        # cleanup intermediate density fields
        del self.mesh_delta

        # compute the “official” mesh_psi as before
        self.mesh_psi = self._compute_psi()
        del self.mesh_delta_real

        # return the per-iteration ψ lists
        return iter_psis

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

        # apply RSD removal in Fourier space
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
            # local LOS: 6 FFTs for ∂i∂j ϕ ˆrᵢˆrⱼ
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

        # Refresh Fourier density so shifts use the latest real-space estimate
        delta_k = self.mesh_delta_real.r2c().copy() 

        # Initialize an array to store displacement shifts for each particle in the reconstructed data space.
        shifts = np.empty_like(self._positions_rec_data)

        # This list is used to optionally store the intermediate displacement fields 
        psis = []

        # Loop over each spatial dimension (x, y, z) to compute shifts along each axis.
        for iaxis in range(delta_k.ndim):

            # No need to compute psi on axis where los is 0
            if self.los is not None and self.los[iaxis] == 0:
                shifts[:, iaxis] = 0.
                psis.append(np.zeros_like(self.mesh_delta_real.value))
                continue

            # Create a copy of the Fourier-space density field `delta_k` to modify and extract displacement components.
            psi = delta_k.copy()

            # Apply Fourier-space operations to extract displacements along the current axis.
            for kslab, islab, slab in zip(psi.slabs.x, psi.slabs.i, psi.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2  
                slab[...] *= 1j * kslab[iaxis] * mask  

            psi = psi.c2r()

            # Read out displacement shifts at the reconstructed particle positions (`_positions_rec_data`).
            # This interpolates the computed displacement field to find the displacements at the actual particle locations.
            shifts[:, iaxis] = self._readout(psi, self._positions_rec_data)

           
            psis.append(psi)
                
            del psi

        # If `self.los` is not explicitly set, compute a normalized los vector from the particle positions.
        if self.los is None:
            los = utils.safe_divide(self._positions_data, utils.distance(self._positions_data)[:, None])
        else:
            los = self.los

        # Adjust shifts for the first iteration using a correction to remove Redshift-Space Distortion (RSD) effects.
        # This follows Eq. 12 from Burden et al. (2015), which applies a correction factor to improve convergence.
        if self._iter == 0:
            shifts -= self.beta / (1 + self.beta) * np.sum(shifts * los, axis=-1)[:, None] * los

        # ====================================================
        # DEBUG output (no guard needed)
        for ax in range(shifts.shape[1]):
            arr = shifts[:, ax]
            self.log_debug(
                f"[iter {self._iter}] shift axis {ax}: "
                f"min={arr.min():.3e}, max={arr.max():.3e}, std={arr.std():.3e}"
            )

        dot = np.sum(shifts * los, axis=-1)
        self.log_debug(
            f"[iter {self._iter}] dot(shifts, los): "
            f"min={dot.min():.3e}, max={dot.max():.3e}, std={dot.std():.3e}"
        )

        if los.ndim == 1:
            self.log_debug(f"[iter {self._iter}] global los = {los}")
        else:
            sample = los[:5]
            self.log_debug(f"[iter {self._iter}] sample los[0:5] =\n{sample}")
        # ====================================================

        # **New Reconstruction Step**
        # Rather than keeping particle positions static throughout iterations, we iteratively update the reconstructed positions.
        # This step modifies `self._positions_rec_data` by subtracting an RSD-related displacement correction.
        # The correction term scales the computed shifts along the los by a factor `self.f`, refining reconstructed positions.
        self._positions_rec_data = self._positions_data - self.f * np.sum(shifts * los, axis=-1)[:, None] * los

        self._iter += 1
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
        Read displacement at input positions.
        
        For explicit position arrays this returns the same result as the standard
        (IFFT) implementation. When the special string 'data' is passed it returns
        the displacements based on the iterative hybrid internal positions.
        
        Parameters
        ----------
        positions : array of shape (N, 3) or string 'data'
            Cartesian positions. Passing 'data' uses the internally updated positions.
        field : {'disp', 'rsd', 'disp+rsd'}, default 'disp+rsd'
            The desired component, where:
            - 'disp' returns the Zeldovich displacement,
            - 'rsd' returns the redshift-space distortion correction,
            - 'disp+rsd' returns the sum.
        
        Returns
        -------
        shifts : array of shape (N, 3)
            The displacement (or total shift) vectors.
        """
        field = field.lower()
        allowed_fields = ['disp', 'rsd', 'disp+rsd']
        if field not in allowed_fields:
            raise ReconstructionError('Unknown field {}. Choices are {}'.format(field, allowed_fields))

        # If the special string 'data' is passed, use the internal positions.
        if isinstance(positions, str) and positions == 'data':
            # Here we compute displacements using the iterative H-IFFT internal positions.
            shifts = np.empty_like(self._positions_rec_data)
            for iaxis, psi in enumerate(self.mesh_psi):
                shifts[:, iaxis] = self._readout(psi, self._positions_rec_data)
            if field == 'disp':
                return shifts
            rsd = self._positions_data - self._positions_rec_data
            if field == 'rsd':
                return rsd
            # 'disp+rsd': add the RSD correction computed iznternally.
            return shifts + rsd

        # For an explicit positions array, simply use the base implementation.
        # Hybrid and IFFT match.
        return super().read_shifts(positions, field=field)







