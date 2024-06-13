import matplotlib.pyplot as plt
import numpy as np

from scipy.special import kl_div as kl
from scipy.stats import pearsonr as pearson
from scipy.signal import argrelmin, argrelmax

import brighteyes_ism.simulation.PSF_sim as sim
import brighteyes_ism.analysis.Tools_lib as tool
from brighteyes_ism.analysis.APR_lib import ShiftVectors

from . import shift_vectors_minimizer as svm
from . import mag_finder as mag


class GridFinder(sim.GridParameters):
    """
    Class to find the optimal parameters to simulate the PSFs as close as possible to the experimental set.

    Parameters

    ----------
    grid_par : object
        object describing the features of the final image (e.g. number of pixel, lateral and axial pixel size...)

    Attributes
    ----------
    shift : float
        shift vectors of the ISM dataset
    rotation : float
        rotation angle
    mirroring : int
        mirroring factor
    M : float
        magnification factor
    """

    def __init__(self, grid_par: sim.GridParameters = None):
        """
        Parameters
        ----------
        grid_par : object, optional
            object describing the features of the final image (e.g. number of pixel, lateral and axial pixel size...)
        """
        sim.GridParameters.__init__(self)
        if grid_par is not None:
            vars(self).update(vars(grid_par))
        self.shift = None  # um

    def estimate(self, dset, wl_ex, wl_em, na):
        """
        Function to find the optimal parameters to simulate the PSFs as close as possible to the experimental set.

        Parameters
        ----------
        dset : np.ndarray
            ISM dataset
        wl_ex : float
            excitation wavelength
        wl_em : float
            emission wavelength
        na : float
            numerical aperture of the objective lens
        """
        ref = dset.shape[-1] // 2
        usf = 50
        # assuming a dataset with shape (Ny x Nx x Nch), we crop the outer frame of 5 pixels

        shift, _ = ShiftVectors(dset[5:-5, 5:-5, :], usf, ref, filter_sigma=1)
        par = svm.find_parameters(shift)
        self.shift = par[0] * self.pxsizex  # um
        self.rotation = par[1]  # rad
        self.mirroring = par[2]  # +/- 1
        self.M = np.round(
            mag.find_mag(self.shift, wl_ex=wl_ex, wl_em=wl_em, pxpitch=self.pxpitch, pxdim=self.pxdim, na=na))


def psf_width(pxsizex: float, pxsizez: float, Nz: int, simPar: sim.simSettings, spad_size, stack='positive') -> int:

    """
    Function calculating the beam waist along the z-axis and the number of pixel required to have the whole simulated
    PSF in the FOV, minimizing simulation time-complexity

    Parameters
    ----------
    pxsizex : float
        dimension in [nm] of every pixel
    pxsizez : float
        discretization step along the z-axis [nm].
    Nz : int
        number of axial planes to discretize the object
    simPar: object 
        object describing the features of excitation light
    spad_size: float
        dimension of the SPAD array detector
    stack: str
        Here one can choose how to generate the PSF stack ( symmetrically or not with respect to the focal plane).
        The default is 'positive'.

    Returns
    -------
    Nx : int
        number of pixel required to have the whole simulated PSF in the FOV for each axial plane
    """
    if stack == 'positive' or stack == 'negative':
        z = pxsizez * Nz
    else:
        z = pxsizez * (Nz//2)


    M2 = 3

    w0 = simPar.airy_unit/2

    z_r = (np.pi * w0**2 * simPar.n) / simPar.wl
    w_z = w0 * np.sqrt( 1 + (M2 * z / z_r )**2)

    Nx = int(np.round((2 * (w_z + spad_size) / pxsizex)))

    if Nx % 2 == 0:
        Nx += 1

    return Nx


def psf_estimator_from_data(data: np.ndarray, exPar: sim.simSettings, emPar: sim.simSettings, grid: sim.GridParameters,
                            downsample: bool = True, stedPar: sim.simSettings=None, z_out_of_focus: str = 'ToFind',
                            n_photon_excitation: int = 1, stack='symmetrical'):
    """
    Function generating rotated PSFs according to the pinholes distribution of the SPAD array detector. This function
    is retrieving a set of parameters from the ISM dataset itself, necessary to obtain an optimal simulation of the
    PSFs, as close as possible to the experimental set.

    Parameters
    ----------
    data : np.ndarray ( Nx x Ny x Nch )
        ISM dataset
    exPar : sim.simSettings
        object describing the excitation step features
    emPar : sim.simSettings
        object describing the emission parameters features
    grid : sim.GridParameters
        parameters describing features of the final image (e.g. number of pixel, lateral and axial pixel size...)
    downsample : bool, optional
        One can choose if downsample the PSFs simulation to achieve optimal accuracy. The default is True.
    stedPar : sim.simSettings, optional
        object describing the STED step features. The default is None.
    z_out_of_focus : str, optional
        Here one can choose how to find the optimal out-of-focus depth of the background plane. It can be found
        minimizing the correlation/Kullback-Leibler divergence between focal PSF and the PSFs along the z-axis or
        by minimizing the correlation between focal PSF and the PSFs along the z-axis. The default is 'ToFind'.
    n_photon_excitation : int, optional
        number of photons exciting the fluorophores during the excitation step. The default is 1.
    stack : str, optional
        Here one can choose how to generate the PSF stack ( symmetrically or not with respect to the focal plane).
        The default is 'symmetrical'.


    Returns
    -------
    Psf3_f : np.ndarray ( Nz x Nx x Ny x Nch )
        complete PSF for every element of the array detector
    detPsf3_f : np.ndarray ( Nz x Nx x Ny x Nch )
        detection PSF for every element of the array detector
    exPsf3_f : np.ndarray ( Nz x Nx x Ny )
        excitation PSFs

    """

    N = int(np.sqrt(data.shape[-1]))

    # if the depth of the background plane is not set, those lines calculates it by minimizing the correlation between
    # focal PSF and the PSFs along the z-axis.
    if isinstance(z_out_of_focus, str) and z_out_of_focus == 'ToFind':
        # generating a stack of PSFs along the z-axis and calculating the correlation curve to minimize
        pxsizez, _ = find_out_of_focus_from_param(grid.pxsizex, exPar, emPar, mode='KL', stack='positive')
        print(pxsizez)
    else:
        pxsizez = float(z_out_of_focus)
        
    # find rotation, mirroring, and magnification parameters from the data
    grid_simul = GridFinder(grid)
    grid_simul.estimate(data, exPar.wl, emPar.wl, emPar.na)
    if downsample is True:
        ups = find_upsampling(grid_simul.pxsizex, pxsize_sim=int(emPar.airy_unit/100))
    else:
        ups = 1

    # calculate optimal simulation range
    Nx_simul = psf_width(grid_simul.pxsizex, grid_simul.pxsizez, grid_simul.Nz, exPar, grid_simul.spad_size(),
                         stack=stack)

    # calculate upsampled pixel size and number of pixels to have a more precise simulation of the PSFs
    pxsize_simul = grid_simul.pxsizex / ups
    Nx_up = Nx_simul * ups

    grid_simul.Nx = Nx_up
    grid_simul.pxsizex = pxsize_simul
    grid_simul.pxsizez = pxsizez
    grid_simul.N = N

    Psf, detPsf, exPsf = sim.SPAD_PSF_3D(grid_simul, exPar, emPar, n_photon_excitation=n_photon_excitation,
                                         stedPar=stedPar, spad=None, stack=stack)  # upsampled PSFs generation

    # downsampling the PSFs to the original pixel size
    if downsample:
        Psf_ds = tool.DownSample(Psf, ups, order='zxyc')
        detPsf_ds = tool.DownSample(detPsf, ups, order='zxyc')
        exPsf_ds = tool.DownSample(exPsf, ups, order='zxy')
    else:
        Psf_ds = Psf
        detPsf_ds = detPsf
        exPsf_ds = exPsf

    return Psf_ds, detPsf_ds, exPsf_ds


def find_upsampling(pxsize_exp: int, pxsize_sim: int = 4):
    """
    Function to find the optimal upsampling factor to use in the PSFs simulation process.

    Parameters
    ----------
    pxsize_exp : int
        pxsize of the experimental ISM data
    pxsize_sim : int
        Target pixel size for the upsampling factor calculation. The default is 4.

    Returns
    -------
    ups_opt : int
        optimal upsampling factor

    """
    # generating an array containing a set of valid values as up sampling factors
    ups = np.arange(1, np.floor(pxsize_exp)).astype(int)

    l = int(len(ups))

    res = np.empty(l)

    for i in range(l):
        res[i] = (pxsize_exp / ups[i] - pxsize_sim) ** 2 # norm function to find the upsampling value that minimizes
        # the difference between the experimental and simulated pixel size (passed as default in this function)

    index = np.argmin(res)

    ups_opt = ups[index]  # optimal upsampling value retrieved

    return ups_opt


def find_out_of_focus(gridPar: sim.GridParameters, input_psf: list, mode: str = 'Pearson',
                                graph: bool = True):
    """
    Function retrieving the optimal out-of-focus depth of the background plane by minimizing the
    Pearson correlation/Kullback-Leibler divergence between the focal PSF and the PSFs along the z-axis

    Parameters
    ----------
    gridPar : sim.GridParameters
        object describing the features of the final image
    input_psf : list, optional
        List containing the PSF stack, used to calculate Correlation/Divergence and find the out-of-focus depth.
        The default is None.
    mode : str, optional
        Choose how to perform the correlation measure.It can be performed evaluating the Kullback-Leibler divergence or
        the Pearson correlation as metric. The default is 'KL'.
    graph : bool, optional
        One can choose if visualize the PSFs correlation graph along the axial dimension. The default is True.

    Returns
    -------
    optimal_depth : float
        Optimal depth where to pose the out-of-focus plane for the Ml-ISM reconstruction.
    PSF :
        Stack used to calculate the correlation/divergence curve along the axial dimension.
    """

    if input_psf is None:
        raise Exception("PSF is not an input.")

    correlation, PSF = conditioning(gridPar=gridPar, input_psf=input_psf, mode=mode)

    optimal_depth = find_max_discrepancy(correlation=correlation, gridpar=gridPar, mode=mode, graph=graph)

    return optimal_depth, PSF


def find_out_of_focus_from_param(pxsizex: float = None, exPar: sim.simSettings = None, emPar: sim.simSettings = None,
                                 grid: sim.GridParameters = None, stedPar: sim.simSettings=None, mode: str = 'Pearson',
                                 stack: str = 'symmetrical', graph: bool = False):
    """
    Function retrieving the optimal out-of-focus depth of the background plane by minimizing the Pearson
    correlation/Kullback-Leibler divergence between the focal PSF and the PSFs along the z-axis calculated
    standing on the set of parameters passed as input

    Parameters
    ----------
    pxsizex : float, optional
        lateral dimension in [nm] of one image voxel
    exPar : sim.simSettings
        object describing the excitation step
    emPar : sim.simSettings
        object describing the emission parameters
    grid : sim.GridParameters
        parameters describing features of the final image (e.g. number of pixel, lateral and axial pixel size...)
    stedPar : sim.simSettings
        object describing the STED parameters
    mode : str, optional
        Choose how to perform the correlation measure.It can be performed evaluating the Kullback-Leibler divergence or
        the Pearson correlation as metric. The default is 'KL'.
    stack : str, optional
        Here one can choose how to generate the PSF stack (symmetrically or not with respect to the focal plane).
        The default is 'positive'.
    graph : bool, optional
        One can choose if visualize the PSFs correlation graph along the axial dimension. The default is False.


    Returns
    -------
    optimal_depth : float
        Optimal depth where to pose the out-of-focus plane for the Ml-ISM reconstruction.
    PSF :
        Stack used to calculate the correlation/divergence curve along the axial dimension.

    """

    if exPar is None:
        raise Exception("PSF parameters are needed.")

    if emPar is None:
        raise Exception("PSF parameters are needed.")

    if pxsizex is None and grid is None:
        raise Exception("Pixel size is needed as input.")

    if grid is None:
        range_z = 1.5*exPar.depth_of_field
        nz = 60

        gridPar = sim.GridParameters()
        gridPar.Nz = nz
        gridPar.pxsizez = np.round(range_z / nz)
        gridPar.pxsizex = pxsizex
    else:
        gridPar = grid.copy()

    # finding the minimum number of pixels needed to have the simulated PSFs contained in the FOV (in order to
    # minimize the time complexity of the simulation process)

    Nx_temp = psf_width(pxsizex, gridPar.pxsizez, gridPar.Nz, exPar, gridPar.spad_size())
    gridPar.Nx = Nx_temp

    # function calculating the correlation curve with Pearson correlation or Kullback-Leibler divergence
    correlation, PSF = conditioning(gridPar=gridPar, emPar=emPar,
                                    exPar=exPar, stedPar=stedPar, mode=mode,
                                    stack=stack)

    optimal_depth = find_max_discrepancy(correlation=correlation, gridpar=gridPar, mode=mode, graph=graph)

    return optimal_depth, PSF


def find_max_discrepancy(correlation: np.ndarray, gridpar: sim.GridParameters, mode: str, graph: bool):
    """
    Function retrieving the optimal out-of-focus depth from a discrepancy curve.

    Parameters
    ----------
    correlation : np.ndarray
        dicrepancy curve (1D).
    gridpar : object
        parameters describing features of the final image (e.g. number of pixel, lateral and axial pixel size...)
    mode : str, optional
        Choose how to perform the correlation measure.It can be performed evaluating the Kullback-Leibler divergence or
        the Pearson correlation as metric. The default is 'KL'.
    graph : bool, optional
        One can choose if visualize the PSFs correlation graph along the axial dimension. The default is False.

    Returns
    -------
    optimal_depth : float
        Optimal depth where to pose the out-of-focus plane for the Ml-ISM reconstruction.
    PSF :
        Stack used to calculate the correlation/divergence curve along the axial dimension.

    """

    if mode == 'KL':
        idx = np.asarray(argrelmax(correlation)).ravel()[0]  # find maximum of the KL divergence curve
    elif mode == 'Pearson':
        idx = np.asarray(argrelmin(correlation)).ravel()[0]  # find the minimum of the Pearson correlation curve
    else:
        raise Exception("Discrepancy method unknown.")

    optimal_depth = idx * gridpar.pxsizez  # optimal out-of-focus depth retrieved

    # plotting the correlation/divergence curve and the optimal out-of-focus depth
    if graph:
        z = np.arange(0, gridpar.Nz * gridpar.pxsizez, gridpar.pxsizez)
        plt.figure()
        plt.plot(z, correlation)
        plt.plot(optimal_depth, correlation[idx], 'ro')
        plt.vlines(optimal_depth, 0, correlation[idx], linestyles='dashed', label='optimal_depth')

    return optimal_depth


def conditioning(gridPar: sim.GridParameters, exPar: sim.simSettings = None, emPar: sim.simSettings = None,
                 stedPar: sim.simSettings = None, mode='Pearson', stack='positive', input_psf=None):
    """
    Function calculating the correlation curve with Pearson correlation or Kullback-Leibler divergence

    Parameters
    ----------
    gridPar : sim.GridParameters
        object describing the features of the final image (e.g. number of pixel, lateral and axial pixel size...)
    exPar : sim.simSettings
        object describing the excitation step features
    emPar : sim.simSettings
        object describing the emission parameters features
    stedPar : sim.simSettings
        object describing the STED parameters
    mode : str, optional
        Choose how to perform the correlation measure.It can be performed evaluating the Kullback-Leibler divergence or
        the Pearson correlation as metric. The default is 'KL'.
    stack : str, optional
        Here one can choose how to generate the PSF stack (symmetrically or not with respect to the focal plane).
        The default is 'positive'.
    input_psf : list, optional
        List containing the PSF stack, used to calculate Correlation/Divergence and find the out-of-focus depth.
        The default is None.


    Returns
    -------
    correlation : np.ndarray
        Correlation/divergence curve along the axial dimension.
    PSF : list
        List containing the PSF stack, used to calculate Correlation/Divergence and find the out-of-focus depth.
    """

    if input_psf is None:
        if exPar is None or emPar is None:
            raise Exception("PSF is not an input. PSF parameters are needed.")

        gridPar.Nx = psf_width(gridPar.pxsizex, gridPar.pxsizez, gridPar.Nz, exPar, gridPar.spad_size())
        # finding the minimum number of pixels needed to have the simulated PSFs contained in the FOV
        # (in order to minimize the time complexity of the simulation process)
        
        # Simulating PSFs stack
        PSF, detPSF, exPSF = sim.SPAD_PSF_3D(gridPar, exPar, emPar, stedPar=stedPar, spad=None, stack=stack)

        # calculating crop value taking into account that PSFs inherently contain a shift as we move apart from the
        # central pixel of the detector array

        npx = int(np.round(((gridPar.N // 2) * gridPar.pxpitch + gridPar.pxdim / 2) / gridPar.M / gridPar.pxsizex))

        PSF = tool.CropEdge(PSF, npx, edges='l,r,u,d', order='zxyc')
        detPSF = tool.CropEdge(detPSF, npx, edges='l,r,u,d', order='zxyc')
        exPSF = tool.CropEdge(exPSF, npx, edges='l,r,u,d', order='zxy')
    else:
        PSF, detPSF, exPSF = input_psf

    # Normalizing PSF of each axial plane with respect to the total flux of each axial plane
    for i in range(gridPar.Nz):
        PSF[i] /= np.sum(PSF[i])

    corr = np.empty(gridPar.Nz)
    # calculating the correlation/divergence between the in-focus PSF and the PSFs at different planes
    if mode == 'KL':
        for i in range(gridPar.Nz):
            corr[i] = kl(PSF[0, ...].flatten(), PSF[i, ...].flatten()).sum()

    elif mode == 'Pearson':
        for i in range(gridPar.Nz):
            corr[i] = pearson(PSF[0, ...].flatten(), PSF[i, ...].flatten())[0]

    return corr, [PSF, detPSF, exPSF]


def combine_psf_irf(psf: np.ndarray, irf: np.ndarray):
    """
        Function calculating the correlation curve with Pearson correlation or Kullback-Leibler divergence

        Parameters
        ----------
        psf : np.ndarray
            4-dimensional array of the spatial PSFs (Nz x Ny x Nx x Nch)
        irf : np.ndarray
            2-dimensional array of the temporal IRFs (Nt x Nch)

        Returns
        -------
        psf_irf : np.ndarray
            5-dimensional array of the spatio-temporal PSFs (Nz x Ny x Nx x Nt x Nch)
        """

    shape_in = psf.shape
    nbin = irf.shape[0]
    shape_out = shape_in + (nbin,)

    psf_rep = np.repeat(psf, nbin).reshape(shape_out)
    psf_rep = np.swapaxes(psf_rep, -2, -1)

    psf_irf = np.einsum('...lm , lm -> ...lm', psf_rep, irf)

    return psf_irf
