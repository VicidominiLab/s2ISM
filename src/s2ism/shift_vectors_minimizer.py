import numpy as np
from scipy.optimize import minimize


def shift_matrix() -> np.ndarray:
    """
    Function returning phantom shift vectors of the 3x3 central ring.

    Returns
    -------
    shift_theor : np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring.

    """

    shift_theor = np.zeros((3, 3, 2))

    for i, n in enumerate(np.arange(-1, 2)):
        for j, m in enumerate(np.arange(-1, 2)):
            shift_theor[i, j, :] = [-n, -m]

    return shift_theor.reshape(9, 2)


def rotation_matrix(theta: float)-> np.ndarray:
    """
    Function returning a 2D rotation matrix

    Parameters
    ----------
    theta : float
        rotation angle in radians.

    Returns
    -------
    rot : np.ndarray
        2 x 2 rotation matrix.
    """
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    rot = np.squeeze(rot_matrix)

    return rot


def mirror_matrix(alpha):

    '''
    Function returning a 2D mirroring matrix
    Parameters
    ----------
    alpha : float

    Returns
    -------
    mirror : np.ndarray
        2 x 2 mirroring matrix.

    '''
    mirror = np.array([[1, 0], [0, alpha]])

    return mirror


def crop_shift(shift_exp: np.ndarray) -> np.ndarray:
    """
    Function cropping the 3x3 central ring of the shift vectors array

    Parameters
    ----------
    shift_exp : np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring.

    Returns
    -------
    shift_cropped : np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring.
    """
    nch = 3
    n = int(np.sqrt(shift_exp.shape[0]))
    shift_exp = shift_exp.reshape(n, n, -1)
    shift_cropped = np.zeros((nch, nch, 2))

    for i, l in enumerate(np.arange(-1, 2)):
        for j, k in enumerate(np.arange(-1, 2)):
            shift_cropped[i, j, :] = shift_exp[n // 2 + l, n // 2 + k, :]

    return shift_cropped.reshape(nch**2, 2)


def transform_shift_vectors(param, shift):
    """
    Function returning the theoretical shift vectors after dilatation and rotation operators

    Parameters
    ----------
    param : list
        list containing the magnification factor and the rotation angle.
    shift : np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring.

    Returns
    -------
    shift_tt : np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring after dilatation and rotation operators.
    """
    a = param[0]
    r = rotation_matrix(param[1])
    m = mirror_matrix(param[2])
    TransformMatrix = a * r @ m
    shift_transf = np.einsum('ij,kj -> ki', TransformMatrix, shift)

    return shift_transf


def loss_shifts(x0, shift_exp: np.ndarray, shift_theor: np.ndarray, mirror: float) -> float:
    """
    Function returning the loss between experimental and theoretical shift vectors

    Parameters
    ----------
    x0 : list
        list containing the magnification factor and the rotation angle.
    shift_exp : np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring.
    shift_theor : np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring.
    mirror : float
        mirror parameter.

    Returns
    -------
    loss : float
        loss between experimental and theoretical shift vectors.
    """
    parameters = [*x0, mirror]
    shift_fin = transform_shift_vectors(parameters, shift_theor)
    loss_func = np.linalg.norm(shift_exp - shift_fin) ** 2

    return loss_func


def loss_minimizer(shift_m, shift_t, alpha_0, theta_0, tol, opt, mirror):
    """
    Function minimizing the loss between experimental and theoretical shift vectors

    Parameters
    ----------
    shift_m : np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring.
    shift_t : np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring.
    alpha_0 : float
        starting point for the dilation parameter
    theta_0 : float
        starting point for the rotation parameter
    tol : float
        tolerance for the minimization
    opt : dict
        dictionary containing the maximum number of iterations for the minimization
    mirror : float
        mirror parameter.

    Returns
    -------
    alpha : float
        dilatation factor describing the dilatation operator
    theta : float
        rotation angle describing the rotation operator
    mirror : float
        mirror parameter.

    """
    results = minimize(loss_shifts, x0 = (alpha_0, theta_0), args=(shift_m, shift_t, mirror), options=opt,
                       tol=tol, method='Nelder-Mead')
    if results.success:
        alpha = results.x[0]
        theta = results.x[1]

        if alpha < 0:
            alpha = abs(alpha)
            theta += np.pi

        return alpha, theta, mirror

    else:
        print('Minimization did not succeed.')
        print(results.message)

        alpha = results.x[0]
        theta = results.x[1]

        if alpha < 0:
            alpha = abs(alpha)
            theta += np.pi

        return alpha, theta, mirror


def find_parameters(shift_exp: np.ndarray, alpha_0: float = 2, theta_0: float = 0.5):
    """
    Function returning the parameters describing the dilatation and rotation operators

    Parameters
    ----------
    shift_exp: np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring.
    alpha_0: float
        starting point for the magnification parameter
    theta_0: float
        initialization for the rotation parameter

    Returns
    -------
    alpha : float
        magnification factor describing the dilatation operator
    theta : float
        rotation angle describing the rotation
    mirror : float
        mirror parameter.
    """
    shift_crop = crop_shift(shift_exp)
    shift_theor = shift_matrix()
    tol = 1e-6
    opt = {'maxiter': 10000}
    params = loss_minimizer(shift_crop, shift_theor, alpha_0, theta_0, tol, opt, mirror=1)
    params_mirror = loss_minimizer(shift_crop, shift_theor, alpha_0, theta_0, tol, opt, mirror=-1)

    Loss_0 = loss_shifts(params, shift_crop, shift_theor, 1)
    Loss_1 = loss_shifts(params_mirror, shift_crop, shift_theor, -1)

    if Loss_0 < Loss_1:
        alpha = params[0]
        theta = params[1]
        mirror = 1
    else:
        alpha = params_mirror[0]
        theta = params_mirror[1]
        mirror = -1

    return alpha, theta, mirror


def calc_shift_vectors(parameters):
    """
    Function returning the theoretical shift vectors after dilatation and rotation operators

    Parameters
    ----------
    parameters : list
        list containing the magnification factor and the rotation angle.

    Returns
    -------
    shift_vectors_array : np.ndarray
        9 x 2 array containing the shift vectors of the 3x3 central ring after dilatation and rotation operators.
    """
    shift_theor = shift_matrix()
    shift_array = transform_shift_vectors(parameters, shift_theor)

    return shift_array
