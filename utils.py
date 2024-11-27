import numpy as np
import random
import torch
import math
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from einops import rearrange, reduce
from tifffile import imsave


def single_depatchify(patched_image: torch.Tensor, patch_size: int, res_x: int = 16, res_y: int = 16) -> torch.Tensor:
    """
    Convert patched image back to the original image resolution.
    """
    return rearrange(patched_image, '(s1 s2) c h w -> c (s1 h) (s2 w)', s1=res_y, s2=res_x, h=patch_size, w=patch_size)


def batch_patchify(batch_data: torch.Tensor, patch_size: int, stride_size: int) -> torch.Tensor:
    """
    Create patches from a batch of images.
    """
    unfolded = batch_data.unfold(2, patch_size, stride_size).unfold(3, patch_size, stride_size)
    return rearrange(unfolded, 'b c s1 s2 h w -> (b s1 s2) c h w')
  
  
def upsample_array(input_array: np.ndarray, res: list[int], channels: int) -> dict[int, np.ndarray]:
    """
    Upsample an input array to multiple resolutions.
    """
    assert input_array.shape[2] == channels, f"Input must have {channels} channels."
    return {r: cv2.resize(input_array, dsize=(r, r)) for r in res}
  
  
def upsample_normal(input_normal, res):
    return upsample_array(input_normal, res, 3)


def upsample_img(input_img, res):
    return upsample_array(input_img, res, 1)


def image_multiscale_resize(
    orig_image: np.ndarray, other_res: list[int] = [64, 128], patch_size: int = 16
) -> tuple[dict[int, np.ndarray], dict[int, torch.Tensor]]:
    """
    Create multiscale versions of the image and their patched tensors.
    """
    images, images_tensors = {}, {}
    orig_res = orig_image.shape[0]

    # Normalize input image format
    if orig_image.shape[0] <= 3:  # Assume C x H x W
        orig_image = np.moveaxis(orig_image, 0, -1)[:, :, 0]
    images[orig_res] = orig_image

    # Rescale to other resolutions
    for res in other_res:
        images[res] = cv2.resize(orig_image, dsize=(res, res))

    dim = 1 if orig_image.ndim == 2 else orig_image.shape[2]
    assert dim in [1, 3], f"Image channels should be 1 or 3, got {dim}."

    # Patchify rescaled images
    for res in other_res + [orig_res]:
        img_tensor = torch.tensor(images[res]).unsqueeze(0).repeat(1, dim, 1, 1)
        images_tensors[res] = batch_patchify(img_tensor, patch_size, patch_size)

    return images, images_tensors


def simple_mean_fusion(out1, out2, out3, normalize = True, e = 1e-6):
    '''
    Result Fusion
    '''
    p1, q1 = out1[:, :, 0] / (out1[:, :, 2] + e), out1[:, :, 1] / (out1[:, :, 2]+ e)
    p2, q2 = out2[:, :, 0] / (out2[:, :, 2]+ e), out2[:, :, 1] / (out2[:, :, 2]+ e)
    p3, q3 = out3[:, :, 0] / (out3[:, :, 2]+ e), out3[:, :, 1] / (out3[:, :, 2]+ e)

    h, w = p1.shape
    p2, q2 = cv2.resize(p2, (h, w)), cv2.resize(q2, (h, w))
    p3, q3 = cv2.resize(p3, (h, w)), cv2.resize(q3, (h, w))

    p_avg = (p1 + p2 + p3) / 3.
    q_avg = (q1 + q2 + q3) / 3.

    result = np.ones((h, w, 3))
    result[:, :, 0] = p_avg; result[:, :, 1] = q_avg
    
    if normalize:
        result = normalize_normals_torch(torch.tensor(result))
        result = result.numpy()
        
    result[out1[:, :, 2] < -0.1] = [-1, -1, -1]
    return result
  

def visualize_patch_normal_pred(normal_patch, res, p_size = 16, save_name = None):
    side_num = int(res / p_size)
    normal = single_depatchify(normal_patch, p_size, side_num, side_num)
    normal_reshape = np.moveaxis(normal, 0, -1)
    plt.imshow((1 + normal_reshape) * 0.5)
    if save_name != None:
        plt.savefig(save_name)
    
def visualize_patch_normal_pred_multiple(normal_patches, res_ls, p_size = 16, save_name = None, result_fusion = True):
    images_num = len(normal_patches) 
    if result_fusion:
        images_num += 1
    fig, axes = plt.subplots(1, images_num, figsize = (images_num * 3, 3))
    last_preds = []
    for i in range(len(normal_patches)):
        side_num = int(res_ls[i] / p_size)
        normal = single_depatchify(normal_patches[i], p_size, side_num, side_num)
        normal_reshape = np.moveaxis(normal, 0, -1)
        if i >= len(normal_patches) - 3:
            last_preds.append(normal_reshape)
            
        axes[i].imshow((1 + normal_reshape) * 0.5)
        axes[i].set_title("Step " + str(i) + " Res " + str(res_ls[i]))
        
    if result_fusion: # smooth over last three predictions
        assert len(last_preds) == 3
        final = simple_mean_fusion(last_preds[-1], last_preds[-2], last_preds[-3], True)
        axes[-1].imshow((1 + final) * 0.5)
        axes[-1].set_title("Combined final result")
    
    plt.tight_layout()
    if save_name!= None:
        plt.savefig(save_name)
    
    return final
    

def compute_mae_mask(predicted_normals, ground_truth_normals, mask):
    cos_similarity = np.sum(predicted_normals * ground_truth_normals, axis=-1)
    angular_diff = np.arccos(np.clip(cos_similarity, -1, 1))
    # cast nans to 0. only sum up loss in masked region
    angular_diff[np.isnan(angular_diff)] = 0
    mae = np.sum(angular_diff * mask) / mask.sum()
    mae = mae / math.pi * 180
    # compute median error
    filtered_arr = angular_diff[mask]
    median_ae = np.median(filtered_arr) / math.pi * 180
    return mae, median_ae, angular_diff


def save_tiff_grayscale(arr, save_name):
    arr = arr.astype(np.float32)
    imsave(save_name, arr)
    
    
# from https://github.com/prs-eth/Marigold/blob/4e7cfb0b0082cdf78b9b36d18b8d79ee13175348/marigold/util/seed_all.py#L26    
def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def normalize_normals_torch(normals):
    if normals.shape[1] == 3:
      norm = torch.sqrt(torch.sum(normals**2, axis=1, keepdims=True))
    else:
      norm = torch.sqrt(torch.sum(normals**2, axis=-1, keepdims=True))
    unit_normals = normals / norm
    return unit_normals


def normalize_normals(normals):
    assert normals.shape[-1] == 3
    norm = np.sqrt(np.sum(normals**2, axis=-1, keepdims=True))
    unit_normals = normals / norm
    return unit_normals


# Convert unit vector normal field to unnormalized -> (-fx, -fy, 1)
def normal_vector_to_fxfy(ns):
    Zx = -ns[:, :, 0] / ns[:, :, 2]
    Zy = -ns[:, :, 1] / ns[:, :, 2]
    return Zx, Zy


def field_convert_to_casorati_C(image, scale = 1):
    #ns = rearrange(image[0], 'c h w -> h w c')
    Zx, Zy = normal_vector_to_fxfy(image)

    dxy,dxx = torch.gradient(Zx)
    dyy,dxy = torch.gradient(Zy)
    dxx,dxy,dyy = scale * dxx, scale * dxy, scale * dyy

    R = 0.5 * (dxx - dyy)
    S = dxy
    T = 0.5 * (dxx + dyy)
    C = torch.sqrt(R**2 + S**2 + T**2)
    W = torch.sqrt(R**2 + S**2)

    Sigma = torch.arctan(T / W)

    return C, Sigma


def gauss(sigma, window_size_coeff = 2):
    w = window_size_coeff * np.floor(5 * sigma / 2) + 1
    x, y = np.meshgrid(np.arange(-(w-1)/2, (w-1)/2 + 1),
                        np.arange(-(w-1)/2, (w-1)/2 + 1))
    r2 = x**2 + y**2
    gaussian = (np.exp(-r2 / (2 * sigma**2))) / ( 2 * math.pi * sigma**2)
    gaussian /= gaussian.sum()
    return gaussian


def gauss_first(sigma, window_size_coeff):
    w = window_size_coeff * np.floor(5 * sigma / 2) + 1
    x, y = np.meshgrid(np.arange(-(w-1)/2, (w-1)/2 + 1),
                        np.arange(-(w-1)/2, (w-1)/2 + 1))
    r2 = x**2 + y**2
    gaussian = (np.exp(-r2 / (2 * sigma**2))) / ( 2 * math.pi * sigma**2)

    gaussian_dx = -x / (sigma**2)
    gaussian_dy = -y / (sigma**2)

    dx = gaussian * gaussian_dx
    dy = gaussian * gaussian_dy
    return dx, dy


def gauss_second(sigma, window_size_coeff):
    w = window_size_coeff * np.floor(5 * sigma / 2) + 1
    x, y = np.meshgrid(np.arange(-(w-1)/2, (w-1)/2 + 1),
                        np.arange(-(w-1)/2, (w-1)/2 + 1))
    r2 = x**2 + y**2
    gaussian = (np.exp(-r2 / (2 * sigma**2))) / ( 2 * math.pi * sigma**2)
    # we normalize the gaussian kernel to have sum 1
    #normalization = np.sum(gaussian.ravel())
    #gaussian /= normalization

    gauss_dxx = (-1 / sigma**2) + (x**2 / sigma**4)
    gauss_dyy = (-1 / sigma**2) + (y**2 / sigma**4)
    gauss_dxy = (-x / sigma**2) * (-y / sigma**2)

    dxx = gaussian * gauss_dxx
    dxy = gaussian * gauss_dxy
    dyy = gaussian * gauss_dyy
    return dxx, dxy, dyy


# assume defualt y-axis direction (positive point down)
def gaussDnew(im,ord,bnd,sig,scl, window_size_coeff=4): #im is a SQUARE matrix. ord is 5 or 6.
    if sig==0:
        dy, dx  = np.gradient(im)
        dxy,dxx = np.gradient(dx)
        dyy,dxy = np.gradient(dy)
        dx,dy,dxx,dxy,dyy = scl*dx,scl*dy,scl**2*dxx,scl**2*dxy,scl**2*dyy
    else:
        Gx, Gy = gauss_first(sig, window_size_coeff)
        Gxx, Gxy, Gyy = gauss_second(sig, window_size_coeff)
        dx  = ndi.convolve(im, Gx)  # x  derivative
        dy  = ndi.convolve(im, Gy)  # y  derivative
        dxx = ndi.convolve(im, Gxx)  # xx derivative
        dxy = ndi.convolve(im, Gxy)  # xy derivative
        dyy = ndi.convolve(im, Gyy)  # yy derivative
        dx,dy,dxx,dxy,dyy = scl*dx,scl*dy,scl**2*dxx,scl**2*dxy,scl**2*dyy

    if ord==6: # for Ivec, append the original value at the beginning
        if sig != 0:
            im = ndi.convolve(im, gauss(sig, window_size_coeff))
        return np.asarray((im,dx,dy,dxx,dxy,dyy))
    return np.asarray((dx,dy,dxx,dxy,dyy))
