import torch
from torch.autograd import grad
import torch.optim as optim
import torch.linalg as tla
import numpy as np
import torch.nn.functional as F

from scheduler import *
from diffusion import cosine_beta_schedule


def setup_scheduler_coeffs(total_timesteps):
    # define beta schedule
    betas = cosine_beta_schedule(timesteps=total_timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    return {
      "betas": betas,
      "alphas": alphas,
      "alphas_cumprod": alphas_cumprod,
      "alphas_cumprod_prev": alphas_cumprod_prev,
      "sqrt_recip_alphas": sqrt_recip_alphas,
      "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
      "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
  }

TIMESTEPS = 300
scheduler_coeffs = setup_scheduler_coeffs(TIMESTEPS)
sqrt_alphas_cumprod = scheduler_coeffs['sqrt_alphas_cumprod']
sqrt_one_minus_alphas_cumprod = scheduler_coeffs['sqrt_one_minus_alphas_cumprod']

# Guidance loss functions
def mask_anchor_loss(image: torch.Tensor, seg_mask: torch.Tensor, gt_val: float = -1.0) -> torch.Tensor:
    """
    Computes loss for masked anchor regions.
    """    
    image = rearrange(image[0], "c h w -> h w c")
    gt = torch.ones_like(image) * gt_val
    gt = gt.to(image.device)

    image_pred = image[seg_mask]
    gt_anchor = gt[seg_mask]
    return torch.abs(image_pred - gt_anchor).mean()


def normalize_normals_torch(normals: torch.Tensor) -> torch.Tensor:
    if normals.shape[1] == 3:
      norm = torch.sqrt(torch.sum(normals**2, axis=1, keepdims=True))
    elif normals.shape[0] == 3:
      norm = torch.sqrt(torch.sum(normals**2, axis=0, keepdims=True))
    else:
      norm = torch.sqrt(torch.sum(normals**2, axis=-1, keepdims=True))
    unit_normals = normals / norm
    return unit_normals


def angle_vector_compute(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
    Computes angles between corresponding vectors in two tensors.
    """  
    angles_prod = torch.einsum('ijk, ijk->jk', vec1, vec2)
    angles_clamp = torch.clamp(angles_prod, min = -1+1e-4, max = 1-1e-4)
    return torch.acos(angles_clamp)
  
  
def simple_angle_loss(image: torch.Tensor, mask: torch.Tensor, patch_size : int = 16) -> tuple:
    _, height, width = image[0].shape

    mask_0 = ((torch.arange(height) + 1) % patch_size == 0)
    mask_1 = ((torch.arange(height) % patch_size == 0))
    mask_0[-1] = False
    mask_1[0] = False

    image = image[0]

    loss_td = _compute_loss(image, mask_0, mask_1, mask)
    loss_lr = _compute_loss(image.permute(1, 0, 2), mask_0, mask_1, mask)

    return loss_td, loss_lr
  

def _compute_loss(image, mask_0, mask_1, mask):
    """
    Helper to compute directional loss (TD/LR).
    """
    norm_0 = torch.linalg.norm(image[:, mask_0], dim=0)
    norm_1 = torch.linalg.norm(image[:, mask_1], dim=0)
    loss = torch.einsum("ijk,ijk->jk", image[:, mask_0] / norm_0, image[:, mask_1] / norm_1)
    loss = torch.clamp(loss, min=-1 + 1e-4, max=1 - 1e-4)
    return torch.acos(loss) * mask[mask_0]
  
  
def linear_angle_loss(image, mask, normalize = False, patch_size = 16):
    """
    Spatial consistency loss to encourage constant curvature (linear normals) along patch seams
    """
    if normalize:
      image = normalize_normals_torch(image)

    _, height, width = image[0].shape
    assert height == width

    mask_0 = ((torch.arange(height) + 1) % patch_size == 0) # location 15, 31, 63, ...
    mask_0_prev = ((torch.arange(height) + 2) % patch_size == 0) # location 14, 30, ...
    mask_1 = ((torch.arange(height) % patch_size == 0)) # location 0, 16, 32, ...
    mask_1_next = ((torch.arange(height) - 1) % patch_size == 0) # location 1, 17, 33, ...

    mask_0[-1] = False; mask_0_prev[-2] = False; mask_1[0] = False; mask_1_next[1] = False # corner/edge cases
    image = image[0]

    angle_interp_next_td = image[:, mask_0] + (image[:, mask_0] - image[:, mask_0_prev]) # next vector pred from prev patch
    angle_interp_prev_td = image[:, mask_1] - (image[:, mask_1_next] - image[:, mask_1]) # prev vector pred from next patch
    angle_next_td = angle_vector_compute(angle_interp_next_td / tla.norm(angle_interp_next_td, dim = 0), 
                                         image[:, mask_1] / tla.norm(image[:, mask_1], dim = 0))
    angle_prev_td = angle_vector_compute(angle_interp_prev_td / tla.norm(angle_interp_prev_td, dim = 0), 
                                         image[:, mask_0] / tla.norm(image[:, mask_0], dim = 0))
    loss_td = (angle_next_td + angle_prev_td)/2 * (mask)[mask_0]

    angle_interp_next_lr = image[:, :, mask_0] + (image[:, :, mask_0] - image[:, :, mask_0_prev]) # next vector pred from prev patch
    angle_interp_prev_lr = image[:, :, mask_1] - (image[:, :, mask_1_next] - image[:, :, mask_1]) # prev vector pred from next patch
    angle_next_lr = angle_vector_compute(angle_interp_next_lr / tla.norm(angle_interp_next_lr, dim = 0), 
                                         image[:, :, mask_1] / tla.norm(image[:, :, mask_1], dim = 0))
    angle_prev_lr = angle_vector_compute(angle_interp_prev_lr / tla.norm(angle_interp_prev_lr, dim = 0), 
                                         image[:, :, mask_0] / tla.norm(image[:, :, mask_0], dim = 0))
    loss_lr = (angle_next_lr + angle_prev_lr)/2 * (mask)[:, mask_0]

    return loss_td, loss_lr


def integrability_loss(nx, ny, nz, mask = None):
    """
    Computes integrability loss for normals.
    Inspired by SFT: https://github.com/dorverbin/shapefromtexture
    original implementation uses a different axis direction for ny, need to flip q (negative sign cancels out)
    """
    p = -nx / nz
    q = ny / nz

    p[p < -10] = -10
    p[p > 10] = 10

    q[q < -10] = -10
    q[q > 10] = 10

    pi0j0 = p[:-1, :-1]   # p_{i,j}
    pi1j0 = p[1:,  :-1]   # p_{i+1,j}
    pi0j1 = p[:-1, 1: ]   # p_{i,j+1}
    pi1j1 = p[1:,  1: ]   # p_{i+1,j+1}

    qi0j0 = q[:-1, :-1]   # q_{i,j}
    qi1j0 = q[1:,  :-1]   # q_{i+1,j}
    qi0j1 = q[:-1, 1: ]   # q_{i,j+1}
    qi1j1 = q[1:,  1: ]   # q_{i+1,j+1}

    loss = torch.square(pi0j0 + pi0j1 - pi1j0 - pi1j1 - qi0j0 + qi0j1 - qi1j0 + qi1j1)
    if mask is not None:
      loss = loss[mask]
    return loss


def guidance_DDIM(input_size, batch_img_patch, batch_img_orig, model, input_noise, ds : DDIMScheduler, mask, start_ts = 300, 
                  patch_size = 16, lr = 0.01, guidance_start = 5, guidance_end = 50, int_loss_weight = 1., int_loss_ts = 40):
    patch_num = int(input_size / patch_size)
    batch_num = len(batch_img_patch)

    img = input_noise; imgs = []; pred_x0s = []
    sync_freq = 1; scheduler = ds
    local_to_global = Rearrange("(b s1 s2) c x y -> b c (s1 x) (s2 y)", s1=patch_num, s2=patch_num)

    model = model.eval()
    ts = scheduler.timesteps[scheduler.timesteps <= start_ts]

    for i, t in enumerate(tqdm(ts)):
        img_copy = img.clone().detach()
        time = torch.full((batch_num,), t, dtype=torch.long).cuda()

        # prev i 10 - 49, 3 times
        if (i + 1) % sync_freq == 0 and i >= guidance_start and i <= guidance_end:
            step_num = 3 if t < ds.timesteps[15] else 10

            for j in range(step_num): # defualt 3
                img = img.requires_grad_()
                # predict noise residual
                noise_pred = model(img, time, batch_img_patch[0:batch_num].float().cuda())
                preds = ds.step(noise_pred, t, img)

                x0_guess = preds['pred_original_sample']
                x0_guess_global = local_to_global(x0_guess)

                int_loss = integrability_loss(x0_guess_global[0][0], x0_guess_global[0][1], x0_guess_global[0][2], None).mean()
                loss_td, loss_lr = linear_angle_loss(x0_guess_global.cuda(), mask.cuda())

                boundary_loss = (loss_lr).mean() + (loss_td).mean()
                loss = boundary_loss

                if t <= ds.timesteps[int_loss_ts]: # integrability loss applied at later steps
                  loss = boundary_loss + int_loss * int_loss_weight

                norm_grad = grad(outputs = loss, inputs=img)[0]
                norm_grad.data = torch.clamp(norm_grad.data, -5, +5)

                # adjust lr for final runs
                if t <= ds.timesteps[40]:
                  lr = lr / 1.5

                img_copy = img_copy - lr * norm_grad
                img = img_copy.clone().detach()

        # regular denoising step
        with torch.no_grad():
            noise_pred = model(img_copy.cuda(), time, batch_img_patch[0:batch_num].float().cuda())
            preds = ds.step(noise_pred, t, img_copy, eta = 0) # deterministic denoising
            img = preds['prev_sample']
            imgs.append(img.detach().cpu().numpy())
            pred_x0s.append(preds['pred_original_sample'].detach().cpu().numpy())

    return imgs, pred_x0s