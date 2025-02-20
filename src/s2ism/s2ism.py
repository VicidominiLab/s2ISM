import gc
from collections.abc import Iterable
import warnings
import numpy as np
from tqdm import tqdm
import string
import torch
import torch.nn.functional as torchpad
from torch import real, einsum
from torch.fft import fftn, ifftn, ifftshift

from . import psf_estimator as svr


# def torch_conv_fft(signal, kernel_fft):
#     """
#     It calculates the 2D circular convolution of a real signal with a kernel using the FFT method using pytorch.

#     Parameters
#     ----------
#     signal : torch.Tensor
#         Tensor with dimensions (Nz, Nx, Ny, T) OR (Nx, Ny, T, Ch) to be convolved.
#     kernel_fft : torch.Tensor
#         Kernel with dimension (Nz, Nx, Ny, T, Ch) in the frequency domain of the convolution.

#     Returns
#     -------
#         conv : torch.Tensor
#             Circular convolution of signal with kernel.
#     """

#     n_axes = kernel_fft.ndim - 2
#     conv_axes = tuple(range(1, n_axes + 1))

#     if signal.shape[-1] == kernel_fft.shape[-2]:
#         signal = signal.unsqueeze(-1) # (M, Nx, Ny, T, 1)
#     elif signal.shape[-1] == kernel_fft.shape[-1]:
#         signal = signal.unsqueeze(0) # (1, Nx, Ny, T, Ch)
#     else:
#         raise Exception('The signal must have 3 or 4 dimensions.')

#     conv = fftn(signal, dim=conv_axes) * kernel_fft  # product of FFT
#     conv = ifftn(conv, dim=conv_axes)  # inverse FFT of the product
#     conv = ifftshift(conv, dim=conv_axes)  # Rotation of 180 degrees of the phase of the FFT
#     conv = torch.real(conv)  # Clipping to zero the residual imaginary part

#     return conv

def partial_convolution(signal, psf_fft, dim1='ijk', dim2='jkl', axis='jk'):
    dim3 = dim1 + dim2
    dim3 = ''.join(sorted(set(dim3), key=dim3.index))

    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]

    signal_fft = fftn(signal, dim=axis_list[0])

    conv = einsum(f'{dim1},{dim2}->{dim3}', signal_fft, psf_fft)

    conv = ifftn(conv, dim=axis_list[2])  # inverse FFT of the product
    conv = ifftshift(conv, dim=axis_list[2])  # Rotation of 180 degrees of the phase of the FFT
    conv = real(conv)  # Clipping to zero the residual imaginary part

    return conv


def amd_update_fft(img, obj, psf_fft, psf_m_fft, eps: float):
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


    Returns
    -------
    obj_new : np.ndarray ( Nz x Nx x Ny )
        New estimate of the object.

    """

    # Variables initialization

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    alphabet = string.ascii_lowercase  #Letters from 'a' to 'z'
    n_dim = psf_fft.ndim  #dimension of the PSF 
   
    str_psf = alphabet[:n_dim]  #Takes the first n_dim letters of alphabet Es. 'abcde'
    str_first = alphabet[:n_dim-1] #Takes the first n_dim-1 letters of alphabet Es. 'abcd'
    str_second = alphabet[1:n_dim] #Takes the letters from the second to n_dim of alphabet Es. 'bcde'
    axis_str = alphabet[1:n_dim-1] #Common axis on which to perform convolution Es. 'bcd'
    
    # Update
    img_estimate = partial_convolution(obj, psf_fft, dim1=str_first, dim2=str_psf, axis=axis_str).sum(0)
    fraction = torch.where(img_estimate < eps, 0, img / img_estimate)
    del img_estimate
    update = partial_convolution(psf_m_fft, fraction, dim1=str_psf, dim2=str_second, axis=axis_str).sum(-1)
    del fraction
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

    if torch.cuda.is_available() and process == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        process = 'cpu'
        
    
    data = torch.from_numpy(dset * 1.0).type(torch.float32).to(device)
    h = torch.from_numpy(psf * 1.0).type(torch.float32).to(device)
    

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

    flip_ax = list(np.arange(1, data_check.ndim))
    norm_ax = tuple(np.arange(1, h.ndim))
    try:
        h = h / (h.sum(keepdim=True, axis=norm_ax))
        ht = torch.flip(h, flip_ax)
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            device = torch.device("cpu")
            h = h.type(torch.float64).to(device)
            h = h / (h.sum(keepdim=True, axis=norm_ax))

            ht = torch.flip(h, flip_ax)
            
            process = 'cpu'
            warnings.warn("Warning: The algorithms goes in Out Of Memory with CUDA. /nThe algorithm will run on the CPU.")
        else:
            raise

    O = torch.ones(shape_init).to(device)
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

    flag_cpu = True
    
    # Try to run on gpu
    if process == 'gpu':
        flag_cpu = False
        try:
            h_fft = fftn(h, dim=flip_ax)
            h = h.to(torch.device("cpu"))
            ht_fft = fftn(ht, dim=flip_ax)
            ht = ht.to(torch.device("cpu"))
            amd_update_fft(data_check, O, h_fft, ht_fft, b)
            flag_cpu = False
            print('optimized code on GPU')
        except RuntimeError as e:
            # NOTE: the string may change?
            if "CUDA out of memory. " in str(e):
                flag_cpu = True
                print("Warning: The algorithms goes in Out Of Memory with CUDA. The algorithm will run on the CPU.")
            else:
                raise

    if flag_cpu:
        device = torch.device("cpu")
        process = 'cpu'
    else:
        device = torch.device("cuda:0")
        process = 'gpu'
        
    h_fft = fftn(h.to(device), dim=flip_ax)
    del h
    ht_fft = fftn(ht.to(device), dim=flip_ax)
    del ht   
        
    data_check = data_check.to(device)
    O = O.to(device)
    O_all = O_all.to(device)

    while flag:
        O_new = amd_update_fft(data_check, O, h_fft, ht_fft, b)
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