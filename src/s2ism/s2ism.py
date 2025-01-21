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
    It calculates the circular convolution of a real signal with a real kernel using the FFT method using pytorch.

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


def amd_update(img, obj, psf_fft, psf_m_fft, eps: float, device: str):
    """
    It performs an iteration of the AMD algorithm.

    Parameters
    ----------
    img : np.ndarray
        Input image ( Nx x Ny x Nch ).
    obj : np.ndarray
        Object estimated from the previous iteration ( Nz x Nx x Ny ) .
    psf : np.ndarray
        Point spread function ( Nz x Nx x Ny x Nch ).
    psf_m : np.ndarray
        Point spread function with flipped X and Y axis ( Nz x Nx x Ny x Nch ).
    eps : float
        Division threshold (usually set at the error machine value).
    device : str
        Pytorch device, either 'cpu' or 'cuda:0'.

    Returns
    -------
    obj_new : np.ndarray ( Nz x Nx x Ny )
        New estimate of the object.

    """

    # Variables initialization

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sz_o = obj.shape
    Nz = sz_o[0]

    sz_i = img.shape
    Nch = sz_i[-1]

    szt = [Nz] + list(sz_i)
    den = torch.empty(szt).to(device)

    # Update
    for z in range(Nz):
        for c in range(Nch):
            den[z, ..., c] = torch_conv(obj[z], psf_fft[z, ..., c])
    img_estimate = den.sum(0)

    del den

    fraction = torch.where(img_estimate < eps, 0, img / img_estimate)

    del img_estimate

    up = torch.empty(szt).to(device)

    for z in range(Nz):
        for c in range(Nch):
            up[z, ..., c] = torch_conv(fraction[..., c], psf_m_fft[z, ..., c])
    update = up.sum(-1)

    del up, fraction

    obj_new = obj * update

    return obj_new


def amd_stop(o_old, o_new, pre_flag: bool, flag: bool, stop, max_iter: int, threshold: float,
             tot: float, nz: int, k: int):
    """
    function dealing with the iteration stop of the algorithm

    Parameters
    ----------
    o_old : np.ndarray
        Object obtained at the latter iteration ( Nz x Nx x Ny ).
    o_new : np.ndarray
        Object obtained at the current iteration ( Nz x Nx x Ny ).
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
    nz : int
        number of axial planes of interest.
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

    # calculating photon flux in the focal plane reconstruction at the previous iteration
    int_f_old = (o_old[nz // 2]).sum()

    # calculating the photon flux in the focal plane reconstruction at the current iteration
    int_f_new = (o_new[nz // 2]).sum()

    # calculating the derivative of the photon count function in the focal plane
    d_int_f = (int_f_new - int_f_old) / tot

    # calculating the photon flux in the out-of-focus planes reconstruction at the previous iteration
    int_bkg_old = o_old.sum() - int_f_old

    # calculating the photon flux in the out-of-focus planes reconstruction at the current iteration
    int_bkg_new = o_new.sum() - int_f_new

    # calculating the derivative of the photon count function in the out-of-focus planes
    d_int_bkg = (int_bkg_new - int_bkg_old) / tot

    # controlling if the derivative value is under the threshold. The algorithm derivative has to lye under the
    # threshold for two consecutive iterations to stop.
    if isinstance(stop, str) and stop == 'auto':
        if torch.abs(d_int_f) < threshold:
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

    return pre_flag, flag, torch.Tensor([int_f_new, int_bkg_new]), torch.Tensor([d_int_f, d_int_bkg])


def batch_reconstruction(dset: np.ndarray, psf: np.ndarray, batch_size: list, overlap: int, stop='fixed',
                         max_iter: int = 100, threshold: float = 1e-3, rep_to_save: str = 'last',
                         initialization: str = 'flat', process: str = 'gpu'):
    """

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
                                  process: str = 'gpu'):
    """
    Core function of the algorithm

    Parameters
    ----------
    dset : np.ndarray
        Input image ( Nx x Ny x Nt x Nch ).
    psf : np.ndarray
        Point spread function ( Nz x Nx x Ny x Nt x Nch ). Important : Pass the PSF with his entire shape! If the axial
        dimension is null, pass the PSF as (1 x Nx x Ny x Nt x Nch).
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
    O : np.ndarray ( Nz x Nx x Ny x Nt) if rep_to_save is equal to 'last'
                    ( n_iter x Nz x Nx x Ny x Nt) if rep_to_save is equal to 'all'
        reconstructed object.
    counts : np.ndarray ( k x 2 )
            It describes the number of photons on the focal plane and on the out-of-focus planes for every iteration of
            the algorithm.
    diff : np.ndarray ( k x 2 )
        It describes the derivative of the photon counts on the focal plane and on the out-of-focus planes for every
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

    Nz = h.shape[0]
    shape_data = data_check.shape
    Nx = shape_data[0]
    Ny = shape_data[1]
    shape_init = (Nz,) + shape_data[:-1]
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

    flip_ax = list(np.arange(1, len(data_check.shape)))

    for j in range(Nz):
        h[j] = h[j] / (h[j].sum())

    ht = torch.flip(h, flip_ax)

    b = torch.finfo(torch.float).eps  # assigning the error machine value

    # user can decide how to initialize the object, either with the photon flux of the input image or with
    # a flat initialization

    if initialization == 'sum':
        S = data_check.sum(-1) / Nz
        for z in range(Nz):
            O[z, ...] = S
    elif initialization == 'flat':
        O *= data_check.sum() / Nz / Nx / Ny
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

    # FFT transform on the spatial dimensions of the 2 given PSFs
    h_fft = fftn(h, dim=flip_ax)
    del h
    ht_fft = fftn(ht, dim=flip_ax)
    del ht

    while flag:
        O_new = amd_update(data_check, O, h_fft, ht_fft, b, device=device)

        pre_flag, flag, counts[:, k], diff[:, k] = amd_stop(O, O_new, pre_flag, flag, stop, max_iter, threshold, tot,
                                                            Nz, k)

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