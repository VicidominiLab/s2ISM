from .psf_estimator import combine_psf_irf


def kernel(psf, irf):
    """
        It generates 25 delta-like IRFs centered at the correct time.
        it combines them with the PSFs for s2ISM pre-processing.
    """

    # define 25 dirac delta
    # combine_psf_irf(psf, dirac_comb)

    raise NotImplementedError


def unmixing(dset, kernel):
    """
        It unmixes the result of s2ISM (x, y, time).
    """

    raise NotImplementedError
