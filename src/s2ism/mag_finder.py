import numpy as np
from scipy.special import jv
from scipy.optimize import minimize
from scipy.signal import convolve


def scalar_psf(r, wl, na):
    """
    Scalar PSF for a circular aperture

    Parameters
    ----------
    r : np.ndarray
        radial coordinate.
    wl : float
        wavelength.
    na : float
        numerical aperture.

    Returns
    -------
    psf : np.ndarray
        scalar psf.
    """

    k = 2 * np.pi / wl
    x = k * r * na

    psf = np.ones_like(x) * 0.5
    np.divide(jv(1, x), x, out=psf, where=(x != 0))
    psf = np.abs(psf) ** 2

    return psf


def scalar_psf_det(r, wl, na, pxdim, pxpitch, m):
    """
    Scalar PSF for a circular aperture and amd rectangular pinhole

    Parameters
    ----------
    r : np.ndarray
        radial coordinate.
    wl : float
        wavelength.
    na : float
        numerical aperture.
    pxdim : float
        pixel dimension.
    pxpitch : float
        pixel pitch.
    m : float
        magnification.

    Returns
    -------

    psf_det : np.ndarray
        scalar psf at the detector.
    """

    psf = scalar_psf(r, wl, na)
    pinhole = rect(r - pxpitch / m, pxdim / m)
    psf_det = convolve(psf, pinhole, mode='same')

    return psf_det


def rect(r, d):
    """
    Rectangular function

    Parameters
    ----------
    r : np.ndarray
        radial coordinate.
    d : float
        diameter.

    Returns
    -------
    r : np.ndarray
        rectangular function.
    """
    r = np.where(abs(r) <= d / 2, 1, 0)
    return r / d


def shift_value(m, wl_ex, wl_em, pxpitch, pxdim, na):
    """
    Shift value between experimental and theoretical shift vectors

    Parameters
    ----------
    m : float
        magnification.
    wl_ex : float
        excitation wavelength.
    wl_em : float
        emission wavelength.
    pxpitch : float
        pixel pitch.
    pxdim : float
        SPAD pixel dimension.
    na : float
        numerical aperture.

    Returns
    -------
    shift : float
        shift value.
    """
    airy_unit = 1.22 * wl_em / na  # nm
    pxsize = 0.1  # nm
    range_x = int(airy_unit / pxsize)
    ref = np.arange(-range_x, range_x + 1) * pxsize

    psf_det = scalar_psf_det(ref, wl_em, na, pxdim, pxpitch, m)
    psf_ex = scalar_psf(ref, wl_ex, na)
    psf_conf = psf_det * psf_ex

    shift = ref[np.argmax(psf_conf)]

    return shift


def loss_shift(x, shift_exp, wl_ex, wl_em, pxpitch, pxdim, na):
    """
    Loss function for the minimization of the shift between experimental and theoretical shift vectors

    Parameters
    ----------
    x : float
        magnification variable.
    shift_exp : np.ndarray
        experimental shift vectors.
    wl_ex : float
        excitation wavelength.
    wl_em : float
        emission wavelength.
    pxpitch : float
        pixel pitch [nm].
    pxdim : float
        SPAD pixel dimension [nm].
    na : float
        numerical aperture.

    Returns
    -------
    loss_func : float
        loss function.

    """
    shift_t = shift_value(x, wl_ex, wl_em, pxpitch, pxdim, na)
    loss_func = np.linalg.norm(shift_t - shift_exp)**2
    return loss_func


def loss_minimizer(shift_t, wl_ex, wl_em, pxpitch, pxdim, na, m_0, tol, opt):
    """
    Minimization of the loss function

    Parameters
    ----------
    shift_t : float
        theoretical shift vector.
    wl_ex : float
        excitation wavelength [nm].
    wl_em : float
        emission wavelength [nm].
    pxpitch : float
        pixel pitch [nm].
    pxdim : float
        SPAD pixel dimension [nm].
    na : float
        numerical aperture.
    m_0 : float
        initial magnification.
    tol : float
        tolerance.
    opt : dict
        options for the minimization.

    Returns
    -------
    m : float
        magnification.
    """

    result = minimize(loss_shift, x0 = m_0, args=(shift_t, wl_ex, wl_em, pxpitch, pxdim, na), options=opt,
                      tol=tol, method='Nelder-Mead')

    if not result.success:
        print('Minimization did not succeed.')
        print(result.message)

    return result.x[0]


def find_mag(shift, wl_ex, wl_em, pxpitch, pxdim, na):
    """
    Magnification finder

    Parameters
    ----------
    shift : float
        shift between experimental and theoretical shift vectors.
    wl_ex : float
        excitation wavelength [nm].
    wl_em : float
        emission wavelength [nm].
    pxpitch : float
        pixel pitch [nm].
    pxdim : float
        SPAD pixel dimension [nm].
    na : float
        numerical aperture.

    Returns
    -------
    m : float
        magnification.
    """

    opt = {'maxiter': 10000}
    tol = 1e-6
    m_0 = 500
    m = loss_minimizer(shift, wl_ex, wl_em, pxpitch, pxdim, na, m_0, tol, opt)

    return m
