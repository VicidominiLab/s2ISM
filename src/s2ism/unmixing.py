import numpy as np
import gc
import torch
import torch.nn.functional as torchpad
from collections.abc import Iterable
from tqdm import tqdm
from torch.fft import fftn, ifftn, ifftshift
from scipy.signal import unit_impulse


def estimate_time_shifts(dset):
    """
        It estimates the time shifts for each channel from the dataset.
        
    Parameters
    ----------
    dset : np.ndarray
        ISM dataset (Nx, Ny, Nt, Ch).


    Returns
    -------
    time_shifts : np.ndarray
        the time shifts for each channel.
    """
    
    time_shifts = np.argmax( dset.sum(axis=(0, 1)) , axis=0)

    return time_shifts


def generate_delta_IRFs(time_shifts, Nbin=81, Nch=25):
    """
        It generates 25 delta-like IRFs centered at the time indicated in time_shifts.
        
    Parameters
    ----------
    time_shifts: tuple or 1D np.array of int or ‘mid’

    Returns
    -------
    irf : np.ndarray
        the irf for each channel (Nbin, Nch).
    """
        
    if isinstance(time_shifts, str) and time_shifts != 'mid':
        raise ValueError("time_shifts must be a 1D array or tuple or 'mid'.")    
    
    if time_shifts == 'mid':
        time_shifts = np.array([Nbin//2]*Nch, dtype=int)

    # define 25 dirac delta
    irf = np.asarray(list(map(lambda s: unit_impulse(Nbin, idx=s), time_shifts)), dtype=float).T

    return irf


# def optmized_conv(signal, kernel_fft):
#     """
#     It calculates the 2D circular convolution of a real signal with a kernel using the FFT method using pytorch.

#     Parameters
#     ----------
#     signal : torch.Tensor
#         Tensor with dimensions (Nx, Ny, T) to be convolved.
#     kernel_fft : torch.Tensor
#         Kernel with dimension (M, T) in the frequency domain of the convolution.

#     Returns
#     -------
#         conv : torch.Tensor
#             Circular convolution of signal with kernel.
#     """

#     conv = fftn(signal, dim=(-1)) * kernel_fft  # product of FFT
#     conv = ifftn(conv, dim=(-1))  # inverse FFT of the product
#     conv = ifftshift(conv, dim=(-1))  # Rotation of 180 degrees of the phase of the FFT
#     conv = torch.real(conv)  # Clipping to zero the residual imaginary part

#     return conv

def amd_update(img, obj, irf, eps: float, device: str):
    """
    It performs an iteration of the AMD algorithm.
    M is the number of fluorophores with distinct fluorescence lifetime
    Parameters
    ----------
    img : np.ndarray
        Input image ( Nx x Ny x T ).
    obj : np.ndarray
        Object estimated from the previous iteration ( M x Nx x Ny ) .
    irf : np.ndarray
        Input response function ( M x T ).
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

    sz_o = obj.shape #M x Nx x Ny
    M = sz_o[0]
    sz_i = img.shape #Nx x Ny x T
    

    szt = [M] + list(sz_i) #M x Nx x Ny x T
    den = torch.empty(szt).to(device)

    obj_rep = torch.unsqueeze(obj, -1) #(M x Nx x Ny x 1)
    irf_rep = torch.unsqueeze(torch.unsqueeze(irf, 1),1) #(M x 1 x 1 x T)
    den = obj_rep * irf_rep
    
    del obj_rep
    
    img_estimate = den.sum(0) #sum over the fluorophores component dimension

    del den

    fraction = torch.where(img_estimate < eps, 0, img / img_estimate)

    del img_estimate

    up = irf_rep * fraction
    update_t = up.sum(-1) #sum over the time dimension

    del fraction, irf_rep, up

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


def unmixing(dset, irf, stop='fixed', max_iter: int = 100,
                                  threshold: float = 1e-3, rep_to_save: str = 'last', initialization: str = 'flat',
                                  process: str = 'gpu', denoiser=False):
    """
    Core function of the algorithm. It unmixes the result of s2ISM (Nx, Ny, T).

    Parameters
    ----------
    dset : np.ndarray
        Input image ( Nx x Ny x Nt ).
    irf : np.ndarray
        Point spread function ( M x Nt ). Important : Pass the PSF with his entire shape!
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
    h = torch.from_numpy(irf * 1.).to(device)

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

    M = h.shape[0] # number of fluorophores with distinct fluorescence lifetime
    shape_data = data_check.shape # Nx x Ny x T
    Nx = shape_data[0] # spatial dimension x
    Ny = shape_data[1] # spatial dimension y
    T = shape_data[2] # temporal dimension
    shape_init = (M,) + shape_data[:-1] # shape of the object (M x Nx x Ny)
    O = torch.ones(shape_init).to(device)   # object initialization (M x Nx x Ny)
    crop_pad_t = int((shape_data[-1] - h.shape[-1]) / 2)

    if crop_pad_t > 0:
        pad_array = np.zeros(2 * h.ndim, dtype='int')
        pad_array[2:4] = crop_pad_t
        pad_array = tuple(np.flip(pad_array))

        h = torchpad.pad(h, pad_array, 'constant')
    elif crop_pad_t < 0:
        raise Exception('The IRF is bigger than the input. Warning.')

    
    for m in range(M):
        h[m] = h[m] / (h[m].sum())

    #ht = torch.flip(h, [-1]) #flipped IRF along the temporal axis

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

    # Iterative reconstruction process
    if stop != 'auto':
        total = max_iter
    else:
        total = None
    cont = 0
    pbar = tqdm(total=total, desc='Progress', position=0)

    while flag:
        O_new = amd_update(data_check, O, h, b, device=device)

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

