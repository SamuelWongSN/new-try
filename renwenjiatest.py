import torch
import cv2
from model.tinynet import tinyNet
import numpy as np
import random
from PIL import Image
import math
from torchvision.utils import make_grid
def read_img_(path, n_colors = 3):
    '''
    :param path:
    :param n_channels:
    :return numpy.ndarray:
    '''
    if n_colors == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_colors == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img
def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def img_single2tensor(img):
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img
def to_pil_image(pic, mode=None):
    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))
    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))
    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode
    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'
    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))
    return Image.fromarray(npimg, mode=mode)

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    # tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.detach().numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.detach().numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def tensor2uint(img):
    # img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def gen_kernel(k_size=np.array([25, 25]), sf=4, min_var=0.2, max_var=4, noise_level=0):
    scale_factor = np.array([sf, sf])
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = 0 # -noise_level + np.random.rand(*k_size) * noise_level * 2

    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]
    MU = k_size // 2 - 0.5*(scale_factor - 1) # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel
def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)
def stable_isotropic_gaussian_kernel(sig=2.6, l=21, tensor=False):
    x = sig
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k

def surf(Z, cmap='rainbow', figsize=None,sigma = 0.6):
    plt.figure(figsize=figsize)
    ax3 = plt.axes(projection='3d')
    w, h = Z.shape[:2]
    xx = np.arange(0,w,1)
    yy = np.arange(0,h,1)
    X, Y = np.meshgrid(xx, yy)
    plt.title('sigma = {}'.format(sigma))
    ax3.plot_surface(X,Y,Z,cmap=cmap,title ='sigma')
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap=cmap)
    plt.imsave()
def bicubic_degradation(x, sf=3):
    x = imresize_np(x, scale=1/sf)
    return x
def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = torch.from_numpy(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)
    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)
    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)
    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)
    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)
    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()
    return out_2.numpy()

def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

import matplotlib.pyplot as plt
from scipy import ndimage
import os
HR_path = 'D:/NEU/pengzhijue/div2k/Set14/img026.jpg'
HR_img = read_img_(HR_path,n_colors=3)
HR_img = modcrop(HR_img,scale=4)
k1 = stable_isotropic_gaussian_kernel(sig=0.6, l=15, tensor=False)
k2 = stable_isotropic_gaussian_kernel(sig=1.6, l=15, tensor=False)
k3 = stable_isotropic_gaussian_kernel(sig=2.6, l=15, tensor=False)
HR_resize = bicubic_degradation(HR_img,sf=4)
img_L1 = ndimage.filters.convolve(HR_img, np.expand_dims(k1, axis=2), mode='wrap')
img_L1 = bicubic_degradation(np.float32(img_L1 / 255.),sf = 4)
img_L2 = ndimage.filters.convolve(HR_img, np.expand_dims(k2, axis=2), mode='wrap')
img_L2 = bicubic_degradation(np.float32(img_L2 / 255.),sf = 4)
img_L3 = ndimage.filters.convolve(HR_img, np.expand_dims(k3, axis=2), mode='wrap')
img_L3 = bicubic_degradation(np.float32(img_L3 / 255.),sf = 4)
# print(HR_img.shape)
# print(img_L1.shape)
# img_blur = img_L1
# img_degradation1 = bicubic_degradation(img_blur,sf = 4)
# 图片纹理增强
# img_degradation2 = img_blur[0::4,0::4,...]
"""
saveLRblurpath = 'D:/NEU/pengzhijue/div2k/Set14'
imsave(img_L1*255,os.path.join(saveLRblurpath, 'sig0.6_img026.jpg'))
imsave(img_L2*255,os.path.join(saveLRblurpath, 'sig1.6_img026.jpg'))
imsave(img_L3*255,os.path.join(saveLRblurpath, 'sig2.6_img026.jpg'))

img_L1 = ndimage.filters.convolve(HR_img, np.expand_dims(k1, axis=2), mode='wrap')
img_L1 = bicubic_degradation(np.float32(img_L1 / 255.),sf = 4)
img_L2 = ndimage.filters.convolve(HR_img, np.expand_dims(k2, axis=2), mode='wrap')
img_L2 = bicubic_degradation(np.float32(img_L2 / 255.),sf = 4)
img_L3 = ndimage.filters.convolve(HR_img, np.expand_dims(k3, axis=2), mode='wrap')
img_L3 = bicubic_degradation(np.float32(img_L3 / 255.),sf = 4)
saveLRblurpath = 'D:/NEU/pengzhijue/div2k/Set14'
imsave(img_L1*255,os.path.join(saveLRblurpath, 'sig0.6orginal_img026.jpg'))
imsave_g(img_L1*255,os.path.join(saveLRblurpath, 'sig0.6gray_img026.jpg'))
imsave(img_L2*255,os.path.join(saveLRblurpath, 'sig1.6orginal_img026.jpg'))
imsave_g(img_L2*255,os.path.join(saveLRblurpath, 'sig1.6gray_img026.jpg'))
imsave(img_L3*255,os.path.join(saveLRblurpath, 'sig2.6orginal_img026.jpg'))
imsave_g(img_L3*255,os.path.join(saveLRblurpath, 'sig2.6gray_img026.jpg'))


plt.subplot(2, 3, 1)
plt.title(label='sigma = 0.6')
plt.imshow(k1, cmap='gray')
plt.subplot(2, 3, 2)
plt.title(label='sigma = 1.6')
plt.imshow(k2, cmap='gray')
plt.subplot(2, 3, 3)
plt.title(label='sigma = 2.6')
plt.imshow(k3, cmap='gray')
plt.subplot(2, 3, 4)
plt.imshow(img_L1)
plt.subplot(2, 3, 5)
plt.imshow(img_L2)
plt.subplot(2, 3, 6)
plt.imshow(img_L3)
# surf(k1,sigma =0.6)
# surf(k2,sigma =1.6)
# surf(k3,sigma =2.6)
plt.savefig('D:/NEU/pengzhijue/div2k/Set14/img026_all.jpg')
plt.show()
"""

LR_path1 = 'D:/NEU/pengzhijue/div2k/Set14/sig0.6_img026.jpg'
LR_path2 = 'D:/NEU/pengzhijue/div2k/Set14/sig1.6_img026.jpg'
LR_path3 = 'D:/NEU/pengzhijue/div2k/Set14/sig2.6_img026.jpg'
# model = tinyNet()
batchsize = 1
patch_size = 192
img_L3 = read_img_(LR_path3,n_colors=3)
plt.subplot(1, 3, 1)
plt.title(label='LR image with sigma 2.6')
plt.imshow(img_L3, cmap='gray')

LR_img= np.float32(img_L3 / 255.)
LR_img = modcrop(LR_img,scale=4)
LR_img = cv2.cvtColor( LR_img, cv2.COLOR_RGB2BGR)

H , W , C = LR_img.shape
rnd_h = random.randint(0, max(0, H - patch_size))
rnd_w = random.randint(0, max(0, W - patch_size))
LR_img = LR_img[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
plt.subplot(1, 3, 2)
plt.title(label='random crop LR image')
plt.imshow(img_L3, cmap='gray')

LR_img_tensor = img_single2tensor(LR_img)
LR_img_4tensor = LR_img_tensor.unsqueeze(0)
from model.tinynet import tinyNet
model = tinyNet()
# model_path = 'D:/NEU/pengzhijue/experiment/train/model/model_latest.pt'
# state_dict = torch.load(model_path)
# model.load_state_dict(state_dict,strict=False)
# HR_a,HR_b,HR_c,HR_d,HR_e = model.forward(LR_img_4tensor)
H= model.forward(LR_img_4tensor)
H = H.squeeze()
H = H.detach().numpy()
# HR_img = (np.transpose(HR_img, (1, 2, 0)) * 255.0).round()
H = tensor2uint(H)
plt.subplot(1, 3, 3)
plt.title(label='resunet HR ')
plt.imshow(H, cmap='gray')
# cv2.imshow('img',H)
# cv2.waitKey(0)