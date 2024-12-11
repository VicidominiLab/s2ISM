import gc
from collections.abc import Iterable

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as torchpad
from torch.fft import fftn, ifftn, ifftshift
import brighteyes_ism.analysis.Tools_lib as tool

from . import psf_estimator as svr


def torch_conv(signal, kernel_fft):
    """
    It calculates the 2D circular convolution of a real signal with a kernel using the FFT method using pytorch.

    Parameters
    ----------
    signal : torch.Tensor
        Tensor with N dimensions to be convolved.
    kernel_fft : torch.Tensor
        Kernel in the frequency domain of the convolution. It has the same number of dimensions of the signal.

    Returns
    -------
        conv : torch.Tensor
            Circular convolution of signal with kernel.
    """

    conv = fftn(signal) * kernel_fft  # product of FFT
    conv = ifftn(conv)  # inverse FFT of the product
    conv = ifftshift(conv)  # Rotation of 180 degrees of the phase of the FFT
    conv = torch.real(conv)  # Clipping to zero the residual imaginary part

    return conv

def optmized_conv(signal, kernel_fft):
    """
    It calculates the 2D circular convolution of a real signal with a kernel using the FFT method using pytorch.

    Parameters
    ----------
    signal : torch.Tensor
        Tensor with dimensions (M, Nx, Ny) OR (Nx, Ny, T Ch) to be convolved.
    kernel_fft : torch.Tensor
        Kernel with dimension (M, Nx, Ny, T, Ch) in the frequency domain of the convolution.

    Returns
    -------
        conv : torch.Tensor
            Circular convolution of signal with kernel.
    """

    if signal.dim() == 3:
        signal = signal.unsqueeze(-1).unsqueeze(-1) # (M, Nx, Ny, 1, 1)
    elif signal.dim() == 4:
        signal = signal.unsqueeze(0) # (1, Nx, Ny, T, Ch)
    else:
        raise Exception('The signal must have 3 or 4 dimensions.')

    conv = fftn(signal, dim=(1,2)) * kernel_fft  # product of FFT
    conv = ifftn(conv, dim=(1,2))  # inverse FFT of the product
    conv = ifftshift(conv, dim=(1,2))  # Rotation of 180 degrees of the phase of the FFT
    conv = torch.real(conv)  # Clipping to zero the residual imaginary part

    return conv

def amd_update(img, obj, psf_fft, psf_m_fft, eps: float, device: str):
    """
    It performs an iteration of the AMD algorithm.
    M is the number of fluorophores with distinct fluorescence lifetime
    Parameters
    ----------
    img : np.ndarray
        Input image ( Nx x Ny x T x Nch ).
    obj : np.ndarray
        Object estimated from the previous iteration ( M x Nx x Ny ) .
    psf : np.ndarray
        Point spread function ( M x Nx x Ny x T x Nch ).
    psf_m : np.ndarray
        Point spread function with flipped X and Y axis ( M x Nx x Ny x T x Nch ).
    eps : float
        Division threshold (usually set at the error machine value).
    device : str
        Pytorch device, either 'cpu' or 'cuda:0'.

    Returns
    -------
    obj_new : np.ndarray ( M x Nx x Ny )
        New estimate of the object.

    """

    # Variables initialization

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sz_o = obj.shape
    M = sz_o[0]

    sz_i = img.shape
    Nch = sz_i[-1]
    T = sz_i[-2]

    szt = [M] + list(sz_i)
    den = torch.empty(szt).to(device)

    # Update
    # for m in range(M):
    #     for c in range(Nch):
    #         for t in range(T):
    #             den[m, ..., t, c] = torch_conv(obj[m], psf_fft[m, ..., t, c])
    den = optmized_conv(obj, psf_fft)
    img_estimate = den.sum(0)

    del den

    fraction = torch.where(img_estimate < eps, 0, img / img_estimate)

    del img_estimate

    up = torch.empty(szt).to(device)

    # for m in range(M):
    #     for c in range(Nch):
    #         for t in range(T):
    #             up[m, ..., t, c] = torch_conv(fraction[..., t, c], psf_m_fft[m, ..., t, c])
    up = optmized_conv(fraction, psf_m_fft)
    update = up.sum(-1)
    update_t = update.sum(-1)

    del fraction

    obj_new = obj * update_t

    return obj_new


def amd_stop(o_old, o_new, pre_flag: bool, flag: bool, stop, max_iter: int, threshold: float,
             tot: float, M: int, k: int):
    """
    function dealing with the iteration stop of the algorithm

    Parameters
    ----------
    o_old : np.ndarray
        Object obtained at the latter iteration ( M x Nx x Ny ).
    o_new : np.ndarray
        Object obtained at the current iteration ( M x Nx x Ny ).
    pre_flag : bool
        first alert that the derivative of the photon counts has reached the threshold.
        To stop the algorithm both flags must turn into False.
    flag : bool
        second alert that the derivative of the photon counts has reached the threshold.
    stop : string
        If set to 'auto' the algorithm will stop when the derivative of the photon counts function reaches the desired
        threshold. If set to 'fixed' the algorithm will stop when the maximum number of iterations is reached.
    max_iter : int
        maximum number of iterations for the algorithm.
    threshold : float
        if stop is set to 'auto', when the derivative of the photon counter function reaches this value the algorithm
        halt.
    tot : float
        total number of photons in the ISM dataset.
    M : int
        is the number of fluorophores with distinct fluorescence lifetime.
    k : int
        indexing the current algorithm iteration.

    Returns
    -------
    pre_flag : boolean
        first alert that the derivative of the photon counts has reached the threshold. To stop the algorithm both
        flags must turn into False.
    flag : boolean
        second alert that the derivative of the photon counts has reached the threshold.
    list
        [total number of photons in the focal plane, total number of photons in the out-of-focus planes]
    list
        [derivative of the photons count at the current iteration in the focal plane, derivative of the photons count
        at the current iteration in the out-of-focus planes]

    """

    # calculating photon flux for the reconstructed image of the first species at the previous iteration
    int_m1_old = (o_old[0]).sum()

    # calculating the photon flux for the reconstructed image of the first species at the current iteration
    int_m1_new = (o_new[0]).sum()

    # calculating the derivative of the photon count function for the reconstructed image of the first species
    d_int_m1 = (int_m1_new - int_m1_old) / tot

    # calculating the photon flux in the out-of-focus planes reconstruction at the previous iteration
    int_m2_old = (o_old[1]).sum()

    # calculating the photon flux in the out-of-focus planes reconstruction at the current iteration
    int_m2_new = (o_new[1]).sum()

    # calculating the derivative of the photon count function in the out-of-focus planes
    d_int_m2 = (int_m2_new - int_m2_old) / tot

    # controlling if the derivative value is under the threshold. The algorithm derivative has to lye under the
    # threshold for two consecutive iterations to stop.
    if isinstance(stop, str) and stop == 'auto':
        if torch.abs(d_int_m1) < threshold:
            if not pre_flag:
                flag = False
            else:
                pre_flag = False
        elif k == max_iter:
            flag = False
            print('Reached maximum number of iterations.')
    # if the iteration stop rule il claimed to be fixed, the algorithm stop when the maximum number of iterations is
    # reached, default value is 100.
    elif isinstance(stop, str) and stop == 'fixed':
        if k == max_iter:
            flag = False

    return pre_flag, flag, torch.Tensor([int_m1_new, int_m2_new]), torch.Tensor([d_int_m1, d_int_m2])


def batch_reconstruction(dset: np.ndarray, psf: np.ndarray, batch_size: list, overlap: int, stop='fixed',
                         max_iter: int = 100, threshold: float = 1e-3, rep_to_save: str = 'last',
                         initialization: str = 'flat', process: str = 'gpu'):
    """
    ### TO BE REVISIONATO!!!!!!!!!!!!!

    Parameters
    ----------
    dset : np.ndarray
        ISM dataset shaped as (Nx x Ny x Nt x Nch).
    psf : np.ndarray
        ISM PSF shaped as (Nz x Nx x Ny x Nt x Nch).
    batch_size : list
        List containing the size of the batch in the x and y dimensions.
    overlap : int
         Overlap value between the batches.
    stop: string, optional
           String describing how to stop the algorithm. The default is 'auto'. If set to 'auto' the algorithm will stop
            when the derivative of the photon counts reaches the threshold, if set to 'fixed' the algorithm will stop when
            the maximum number of iterations is reached.
    max_iter: int, optional
             maximum number of iteration. The default is 100.
    threshold: float, optional
              The default is 1e-3. This value is needed when we choose stop = 'auto'. When the derivative of the photon counts
              reaches this value the algorithm halt.
    rep_to_save: iter, optional
                  iterable containing the iteration at which one wants to save the algorithm reconstruction.The default is 'all'.
    initialization : string, optional
                      The default is 'flat'. If set to 'flat' the algorithm will initialize the first iteration with a flat object,
                     if set to 'sum' the algorithm will initialize the first iteration with the ISM dataset.
    process: string, optional
               The default is 'gpu'. If set to 'gpu' the algorithm will run on the GPU if available, if set to 'cpu' the algorithm
                will run on the CPU.

    Returns
    -------
    reconstruction : np.ndarray ( Nz x Nx x Ny x Nt x Nch)
                     reconstructed object.
    """

    wx, wy = batch_size
    reconstruction = np.zeros((psf.shape[0],) + dset.shape[:-1])

    nx = dset.shape[0] // (wx - overlap) + int(dset.shape[0] % (wx - overlap) > 0)
    ny = dset.shape[1] // (wy - overlap) + int(dset.shape[1] % (wy - overlap) > 0)

    k_iter = 1
    n_batch = nx * ny
    for i in range(nx):
        for j in range(ny):
            print(f'Batch {k_iter}/{n_batch}')

            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            overlap_x = overlap if i > 0 else 0
            overlap_y = overlap if j > 0 else 0

            indx = np.s_[i * (wx - overlap_x): (i + 1) * wx - i * overlap_x]
            indy = np.s_[j * (wy - overlap_y): (j + 1) * wy - j * overlap_y]

            crop_dset = dset[indx, indy]

            recon, *_ = max_likelihood_reconstruction(dset=crop_dset, psf=psf, stop=stop,
                                                      max_iter=max_iter, threshold=threshold,
                                                      rep_to_save=rep_to_save,
                                                      initialization=initialization, process=process)

            indx = np.s_[i * (wx - overlap_x) + overlap_x // 2: (i + 1) * wx - i * overlap_x]
            indy = np.s_[j * (wy - overlap_y) + overlap_y // 2: (j + 1) * wy - j * overlap_y]

            reconstruction[:, indx, indy] = recon[:, overlap_x // 2:, overlap_y // 2:]
            k_iter += 1

    return reconstruction


def max_likelihood_reconstruction(dset, psf, stop='fixed', max_iter: int = 100,
                                  threshold: float = 1e-3, rep_to_save: str = 'last', initialization: str = 'flat',
                                  process: str = 'gpu', denoiser=False):
    """
    Core function of the algorithm

    Parameters
    ----------
    dset : np.ndarray
        Input image ( Nx x Ny x Nt x Nch ).
    psf : np.ndarray
        Point spread function ( M x Nx x Ny x Nt x Nch ). Important : Pass the PSF with his entire shape!
    stop : string, optional
        String describing how to stop the algorithm. The default is 'auto'. If set to 'auto' the algorithm will stop
        when the derivative of the photon counts reaches the threshold, if set to 'fixed' the algorithm will stop when
        the maximum number of iterations is reached.
    max_iter : int, optional
        maximum number of iteration. The default is 100.
    threshold : float, optional
        The default is 1e-3. This value is needed when we choose stop = 'auto'. When the derivative of the photon counts
        reaches this value the algorithm halt.
    rep_to_save : iter, optional
        object containing the iteration at which one wants to save the algorithm reconstruction.The default is 'all'.
         If set to 'all' the algorithm will save every iteration, if set to 'last' the algorithm will save only the last
         iteration, if set to a list or array the algorithm will save the desired iterations.
    initialization : string, optional
        The default is 'flat'. If set to 'flat' the algorithm will initialize the first iteration with a flat object,
        if set to 'sum' the algorithm will initialize the first iteration with the ISM dataset.
    process : string, optional
        The default is 'gpu'. If set to 'gpu' the algorithm will run on the GPU if available, if set to 'cpu' the algorithm
        will run on the CPU.

    Returns
    -------
    O : np.ndarray ( M x Nx x Ny) if rep_to_save is equal to 'last'
                    ( n_iter x M x Nx x Ny) if rep_to_save is equal to 'all'
        reconstructed object.
    counts : np.ndarray ( k x 2 )
            It describes the number of photons for the first species and the second species for every iteration of
            the algorithm.
    diff : np.ndarray ( k x 2 )
        It describes the derivative of the photon counts for the first species and the second species for every
        iteration of the algorithm.
    k : int
        iteration in which the algorithm stops.

    """

    # Variables initialization taking into account if the data is spread along the axial dimension or not
    device = torch.device("cuda:0" if torch.cuda.is_available() and process == 'gpu' else "cpu")

    data = torch.from_numpy(dset * 1.).to(device)
    h = torch.from_numpy(psf * 1.).to(device)

    oddeven_check_x = data.shape[0] % 2
    oddeven_check_y = data.shape[1] % 2
    check_x = False
    check_y = False

    data_check = data
    if oddeven_check_x == 0:
        check_x = True
        data_check = data_check[1:]
    if oddeven_check_y == 0:
        check_y = True
        data_check = data_check[:, 1:]

    M = h.shape[0]
    shape_data = data_check.shape
    Nx = shape_data[0]
    Ny = shape_data[1]
    T = shape_data[2]
    Ch = shape_data[3]
    shape_init = (M,) + shape_data[:-2]
    O = torch.ones(shape_init).to(device)

    crop_pad_x = int((shape_data[0] - h.shape[1]) / 2)
    crop_pad_y = int((shape_data[1] - h.shape[2]) / 2)

    if crop_pad_x > 0 or crop_pad_y > 0:
        pad_array = np.zeros(2 * h.ndim, dtype='int')
        pad_array[2:4] = crop_pad_x
        pad_array[4:6] = crop_pad_y
        pad_array = tuple(np.flip(pad_array))

        h = torchpad.pad(h, pad_array, 'constant')
    elif crop_pad_x < 0 or crop_pad_y < 0:
        raise Exception('The PSF is bigger than the image. Warning.')

    flip_ax = list(np.arange(1, len(data_check.shape)-1)) #spatial axes
    
    for m in range(M):
        h[m] = h[m] / (h[m].sum())

    ht = torch.flip(h, flip_ax)

    b = torch.finfo(torch.float).eps  # assigning the error machine value

    # user can decide how to initialize the object, either with the photon flux of the input image or with
    # a flat initialization

    if initialization == 'sum':
        S = data_check.sum((-2,-1)) / M
        for m in range(M):
            O[m, ...] = S
    elif initialization == 'flat':
        O *= data_check.sum() / M / Nx / Ny
    else:
        raise Exception('Initialization mode unknown.')

    k = 0

    counts = torch.zeros([2, max_iter + 1]).to(device)
    diff = torch.zeros([2, max_iter + 1]).to(device)
    tot = data.sum()

    if isinstance(rep_to_save, str) and rep_to_save == 'all':
        size = [max_iter + 1] + list(O.shape)
        O_all = torch.empty(size).to(device)
    elif isinstance(rep_to_save, Iterable):
        l = len(rep_to_save)
        size_b = [l] + list(O.shape)
        O_all = torch.empty(size_b).to(device)

    pre_flag = True
    flag = True

    # PSF normalization axial plane wise, with respect to the flux of each plane

    # Iterative reconstruction process
    if stop != 'auto':
        total = max_iter
    else:
        total = None
    cont = 0
    pbar = tqdm(total=total, desc='Progress', position=0)

    # FFT transform of the 2 given PSFs
    h_fft = fftn(h, dim=flip_ax)
    del h
    ht_fft = fftn(ht, dim=flip_ax)
    del ht

    while flag:
        O_new = amd_update(data_check, O, h_fft, ht_fft, b, device=device)

        pre_flag, flag, counts[:, k], diff[:, k] = amd_stop(O, O_new, pre_flag, flag, stop, max_iter, threshold, tot,
                                                            M, k)

        if isinstance(rep_to_save, Iterable) and not isinstance(rep_to_save, str):
            if k in rep_to_save:
                O_all[cont, ...] = O.clone()
                cont += 1
        elif isinstance(rep_to_save, str) and rep_to_save == 'all':
            O_all[k, ...] = O.clone()

        O = O_new.clone()

        k += 1
        pbar.update(1)
    pbar.close()
    #
    if check_x:
        if rep_to_save == 'last':
            pad_arr = np.zeros(2 * len(O.shape), dtype='int')
            pad_arr[-4] = 1
            O = torchpad.pad(O, tuple(pad_arr), 'constant')
        else:
            pad_arr = np.zeros(2 * len(O_all.shape), dtype='int')
            pad_arr[-6] = 1
            O_all = torchpad.pad(O_all, tuple(pad_arr), 'constant')
    if check_y:
        if rep_to_save == 'last':
            pad_arr = np.zeros(2 * len(O.shape), dtype='int')
            pad_arr[-6] = 1
            O = torchpad.pad(O, tuple(pad_arr), 'constant')
        else:
            pad_arr = np.zeros(2 * len(O_all.shape), dtype='int')
            pad_arr[-8] = 1
            O_all = torchpad.pad(O_all, tuple(pad_arr), 'constant')

    if isinstance(rep_to_save, str) and rep_to_save == 'last':
        obj = O.detach().cpu().numpy()
    else:
        obj = O_all.detach().cpu().numpy()

    counts = counts[:, :k].detach().cpu().numpy()
    diff = diff[:, :k].detach().cpu().numpy()

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    return obj, counts, diff, k


def data_driven_reconstruction(dset: np.ndarray, gridPar, exPar, emPar, rep_to_save: str = 'last',
                               initialization: str = 'flat', max_iter: int = 100, stop='fixed', threshold: float = 1e-3,
                               downsample: bool = True, z_out_of_focus='ToFind'):
    """
 ### TO BE REVISIONATO

     Parameters
     ----------
    dset : np.ndarray ( Nx x Ny x Nch)
        ISM dataset to reconstruct
        warning : the ISM dataset should be preferably featured by odd number of pixels in the lateral dimensions
    gridPar : object
        parameters describing features of the acquired image (e.g. number of pixel, lateral and axial pixel size...)
    exPar : object
        excitation parameters
    emPar : object
        emission parameters
    rep_to_save : bool, optional
            if equal to 'last' the algorithm will save only the last instance of the iterative algorithm, if 'all' the
            algorithm will save every instance of the iterative algorithm, if list or array the algorithm will save the
            desired instances.
    initialization: string
         if flat the first instance of the iterative algorithm is initialized to the total flux of the ISM dataset,
         if sum the first instance of the iterative algorithm is initialized such that the total flux of the ISM dataset
         is equally divided on every object's pixel
    max_iter : int, optional
         maximum number of iterations that the algorithm can perform both in fixed and auto stop mode. The default is
         100.
    stop : str, optional
         string describing how one wants to stop the algorithm . in auto mode the algorithm halt when the derivative of
         the photon counts reach the set threshold.
         In fixed mode the algorithm halt when the iterations hit the maximum number of iterations passed as max_iter.
          The default is 'auto'.
    threshold : float, optional
         If the stop rule is set as auto, the algorithm halt when the photon counts derivative reaches this value.
         The default is 1e-3.
    z_out_of_focus :
         if equal to 'ToFind' the algorithm will find the optimal depth of reconstruction through a correlative
         minimization procedure if passed as a float the algorithm will place at that depth the background plane
         of reconstruction


    Returns
    -------
    O : np.ndarray  ( Nz x Nx x Ny) if rep_to_save is equal to 'last'
                 ( n_iter x Nz x Nx x Ny) if rep_to_save is equal to 'all'
                 ( len(rep_to_save) x Nz x Nx x Ny) if rep_to_save is equal to a list or array
     reconstructed object.
    counts : np.ndarray ( k x 2 )
     number of the photons on the axial planes and on the out-of-focus planes for every iteration of the algorithm.
    diff : np.ndarray ( k x 2 )
     photon counts derivative on the axial planes and on the out-of-focus planes for every iteration of the algorithm.
    k : int
     total number of iterations performed by the algorithm.
    PSF : np.ndarray ( Nz x Nx x Ny x Nch) if PSF is passed as a string and equal to 'blind'
    """

    psf, _, _ = svr.psf_estimator_from_data(dset, exPar, emPar, gridPar, downsample=downsample,
                                            z_out_of_focus=z_out_of_focus)

    obj, counts, diff, k = max_likelihood_reconstruction(dset, psf, stop=stop, max_iter=max_iter, threshold=threshold,
                                                         rep_to_save=rep_to_save, initialization=initialization)

    return obj, counts, diff, k, psf


from scipy.signal import convolve2d
#import bm3d as bm3d


def interpolate_image(x, conv_filter=None):
    if conv_filter is None:
        conv_filter = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    return convolve2d(x, conv_filter, mode='same')


def generate_mask(shape, idx, width=3):
    m = np.zeros(shape)

    phasex = idx % width
    phasey = (idx // width) % width

    m[phasex::width, phasey::width] = 1
    return m


def invariant_denoise(img, width, denoiser, h):
    n_masks = width * width

    interp = interpolate_image(img, h)

    output = np.zeros(img.shape)

    for i in range(n_masks):
        m = generate_mask(img.shape, i, width=width)
        input_image = m * interp + (1 - m) * img
        input_image = input_image.astype(img.dtype)
        output += m * denoiser(input_image)
    return output


from skimage.metrics import mean_squared_error as mse

from skimage import data, img_as_float, img_as_ubyte


def mseh(im_list, ref):
    ref = img_as_float(ref)
    im_list = [img_as_float(x) for x in im_list]

    loss = [mse(x, ref) for x in im_list]

    return loss


def kl(im_list, ref):
    ref = img_as_float(ref)
    im_list = [img_as_float(x) for x in im_list]

    loss = [tool.kl_divergence(x, ref, normalize_entries=True).sum() for x in im_list]

    return loss