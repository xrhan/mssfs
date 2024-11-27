import torch
import pickle
import os
from datetime import datetime
import json
import argparse

from diffusion import *
from utils import *
from ddim import *
from lighting import lighting_guided_DDIM_restart

# Define global variables
LIGHTING_RESTART_TS = 232

def multiscale_guidance(
        d_model, 
        patch_images, 
        orig_images, 
        res_list, 
        ts_list, 
        guidance_list, 
        lr_list, 
        input_noise, 
        scheduler, 
        device, 
        lighting_guidance = False, 
        int_loss_weight = 0.5, 
        p_size = 16, 
        seg_mask = None):
    """
    Perform multiscale guidance with optional lighting adjustments.

    Returns:
        imgs_prev: Final image predictions.
        result_track: Dictionary tracking predictions and intermediate results.
    """    

    # initial run, original resolution
    result_track = {}
    res_init, ts_init, lr_init, g_init = res_list[0], ts_list[0], lr_list[0], guidance_list[0]
    batch_img_patch, subsampled_img = torch.tensor(patch_images[res_init]), torch.tensor(orig_images[res_init])
    mask_init = torch.ones((res_init, res_init))

    imgs, pred_x0s = guidance_DDIM(res_init, batch_img_patch.to(device), subsampled_img.to(device), d_model, 
                                   input_noise.to(device), scheduler, mask_init.to(device), start_ts = ts_init, 
                                   lr = lr_init, guidance_start = g_init, guidance_end = 50, int_loss_weight = int_loss_weight)
    
    result_track[(res_init, 0)] = {'pred_x0s' : pred_x0s, 'imgs': imgs}

    if lighting_guidance[0]:
        flip_restart_ts = LIGHTING_RESTART_TS
        normal_flipped_noisy, _, _ = lighting_guided_DDIM_restart(res_init, batch_img_patch, imgs[-1], subsampled_img, 
                                                                  method = 'lstsq', min_pixels = lighting_pixels_threshold, seg_mask = seg_mask, 
                                                                  flip_restart_ts = flip_restart_ts)
        
        imgs, pred_x0s = guidance_DDIM(res_init, batch_img_patch.to(device), subsampled_img.to(device), d_model, 
                                       normal_flipped_noisy.to(device), scheduler, mask_init.to(device), start_ts = flip_restart_ts, 
                                       lr = lr_init, guidance_start = 0, guidance_end = 50, int_loss_weight = int_loss_weight)
        
        result_track[(res_init, 0.5)] = {'pred_x0s' : pred_x0s, 'imgs': imgs}

    imgs_prev = imgs

    for r in range(1, len(res_list)):
        # guided denoising at certain resolution and learning rate
        res_prev, res_new = res_list[r-1], res_list[r]
        prev_result = single_depatchify(imgs_prev[-1], p_size, int(res_prev / p_size), int(res_prev / p_size))
        normal_res_new = upsample_normal(np.moveaxis(prev_result, 0, -1), [res_new])

        normal_guess = torch.tensor(np.moveaxis(normal_res_new[res_new], -1, 0)).unsqueeze(0) # 1 x c x H x W
        normal_guess_batch = batch_patchify(normal_guess, p_size, p_size) # non-overlapping

        # add random noise back to restart at an earlier timestep
        batch_size = int(res_new / p_size)**2
        back_timestep = ts_list[r]
        lr_new, g_new = lr_list[r], guidance_list[r]
        t = torch.full((batch_size,), back_timestep, dtype=torch.long)
        normal_guess_noisy = q_sample(normal_guess_batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        mask_everything = torch.ones(res_new, res_new)

        batch_img_patch, subsampled_img = torch.tensor(patch_images[res_new]), torch.tensor(orig_images[res_new])
        imgs, pred_x0s = guidance_DDIM(res_new, batch_img_patch.to(device), subsampled_img.to(device), d_model, 
                                       normal_guess_noisy.to(device), scheduler, mask_everything.to(device), start_ts = back_timestep, 
                                       lr = lr_new, guidance_start = g_new, guidance_end = 50, int_loss_weight = int_loss_weight)
        
        # optional lighting guidance
        if lighting_guidance[r]:
            flip_restart_ts = LIGHTING_RESTART_TS
            normal_flipped_noisy, _, _ = lighting_guided_DDIM_restart(res_new, batch_img_patch, imgs[-1], subsampled_img, 
                                                                      method = 'lstsq', min_pixels = lighting_pixels_threshold, seg_mask = None, 
                                                                      flip_restart_ts = flip_restart_ts)
            
            imgs, pred_x0s = guidance_DDIM(res_new, batch_img_patch.to(device), subsampled_img.to(device), d_model, 
                                           normal_flipped_noisy.to(device), scheduler, mask_everything.to(device), start_ts = flip_restart_ts, 
                                           lr = lr_new, guidance_start = 0, guidance_end = 50, int_loss_weight = int_loss_weight)
            result_track[(res_new, 0.5)] = {'pred_x0s' : pred_x0s, 'imgs': imgs}
        
        imgs_prev = imgs
        result_track[(res_new, r)] = {'pred_x0s' : pred_x0s, 'imgs': imgs}

    return imgs_prev, result_track


def run_multiscale_exp(input_noise, 
                       diffusion_model, 
                       input_image_patch, 
                       input_image, 
                       scheduler_config, 
                       device, 
                       random_seed, 
                       save_name = 'exp', 
                       int_loss_weight = 0.5):
    """
    Wrapper function with scheduler info and setup - run multiscale sampling with a specific scheduler configuration.
    """    
    
    TIMESTEPS = 300
    # set_scheduler(timesteps) as global variables
    # define beta schedule
    betas = cosine_beta_schedule(timesteps=TIMESTEPS)
    ds = DDIMScheduler(betas.cuda(), 300, timestep_spacing = "linspace")
    ds.set_timesteps(50) # DDIM with 50 steps

    all_imgs = {}
    all_result_tmp = {}

    res_l, ts_l = scheduler_config['res_list'], scheduler_config['ts_list'], 
    guidance_l, lr_l = scheduler_config['guidance_list'], scheduler_config['lr_list']
    lighting_list = scheduler_config['light_list']
    
    imgs_prev, result_tmp = multiscale_guidance(diffusion_model,input_image_patch,input_image, 
                                                res_l, ts_l, guidance_l, lr_l, 
                                                input_noise,ds, device, 
                                                lighting_list, int_loss_weight)
    
    all_imgs = imgs_prev; all_result_tmp = result_tmp
    res_list = scheduler_config['res_list']
    
    plot_save_name = save_name + '.png'
    to_plot = []
    to_plot_res_list = [] # a sepearate multiscale resolution schedule to account for lighting guidance
    result_tmp = all_result_tmp
    
    for r in range(len(res_list)):
        curr_res_result = result_tmp[(res_list[r], r)]
        img_x0s = curr_res_result['imgs']
        to_plot.append(img_x0s[-1])
        to_plot_res_list.append(res_list[r])
        
        if lighting_list[r]:
            curr_res_result = result_tmp[(res_list[r], 0.5)]
            img_x0s = curr_res_result['imgs']
            to_plot.append(img_x0s[-1])
            to_plot_res_list.append(res_list[r])
    
    final = visualize_patch_normal_pred_multiple(to_plot, to_plot_res_list, 16, plot_save_name, result_fusion=True)
    to_plot.append(final)
    
    img_save_name = save_name + '_img.pkl' # save last iteration result
    with open(img_save_name, 'wb') as pickle_file3:
        pickle.dump(to_plot, pickle_file3)
    
    return all_imgs
          

def run_single_stimuli(image_np_name, save_name, init_noise, random_seed, scheduler_config, device, int_weight, root_dir):
    # load test image
    img_data_path = root_dir + 'test_data/' + image_np_name
    
    test_img = np.load(img_data_path)
    subsampled_images, subsampled_images_patch = image_multiscale_resize(test_img, [32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 240, 256])
    run_multiscale_exp(init_noise, diffusion_model, subsampled_images_patch, subsampled_images, scheduler_config, device, 
                       random_seed, save_name, int_loss_weight=int_weight) # default 0.5
    
    
def set_initial_noise(init_res, seed, channels = 3, image_size = 16):
    random_seed = seed
    seed_all(random_seed)
    total_num = (int(init_res / image_size)) ** 2
    noise_fixed = torch.randn((total_num, channels, image_size, image_size))
    return noise_fixed


def load_model(model_id, root_dir, device):
    # Specify pre-trained model checkpoint
    all_models = {
        1: 'patch_unet_aug_default.ckpt'
    }
    model_name = all_models[model_id]
    model_path = os.path.join(root_dir, 'saved_models', model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint '{model_path}' not found.")

    diffusion_model = Unet(
        dim=16,
        input_channels=4,
        channels=3,
        dim_mults=(1, 2, 4, 8),
        condition=True
    )
    diffusion_model.load_state_dict(torch.load(model_path))
    return diffusion_model.eval().to(device), model_name


def setup_experiment_directory(root_dir, exp_save_name):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")
    folder_name = os.path.join(root_dir, 'eval', f"data_{formatted_datetime}_{exp_save_name}")

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return folder_name


def schedulers_default():
    # you can define other V-cycle or coarse-fine sampling schedulers here
    res_list8 = [160, 128, 64, 80, 96, 112, 128, 144, 160]
    guidance_list8 = [8] + [0] * 8
    lr_list8 = [20, 15, 10, 10, 10, 15, 15, 20, 20]
    ts_list8 = [300] + [232] * 8
    lighting_list8 = [True] * 2 + [False] * 7
    all_schedulers = {}
    all_schedulers[8] = {'res_list':res_list8, 'ts_list':ts_list8, 'guidance_list': guidance_list8, 
                         'lr_list': lr_list8, 'light_list': lighting_list8}
    return all_schedulers


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="multiscale experiment parameters")
    parser.add_argument("--lighting-guidance", type=bool, default=False)
    parser.add_argument("--lighting-pixels-num", type=int, default=10)
    parser.add_argument("--files", type=int, default=1)
    parser.add_argument("--model", type=int, default=3)
    parser.add_argument("--save-name", type=str, default="", help="experiment name")
    parser.add_argument("--int-weight",type=float, default=0.5, help='integrability loss weight')
    
    # Extract arguments
    args = parser.parse_args()
    use_lighting_guidance = args.lighting_guidance
    lighting_pixels_threshold = args.lighting_pixels_num
    file_name_index = args.files
    model_id = args.model
    exp_save_name = args.save_name
    int_weight = args.int_weight
    
    # Define root directory and device
    root_dir = './'
    device = "cuda"
    
    # Load model
    diffusion_model, model_name = load_model(model_id, root_dir, device)
    
    # Set up experiment directory
    folder_name = setup_experiment_directory(root_dir, exp_save_name)

    # Specify test image
    all_img_names0 = ['circles.npy', 'rings.npy', 'snake.npy', 'star.npy']
    all_file_names = {0: all_img_names0}
    all_img_names = all_file_names[file_name_index]
    
    # Specify test seeds
    all_schedulers = schedulers_default()
    all_seeds = np.arange(0, 4)
    
    # save experiment config
    exp_global_config = {
        'pre-trained model': model_name, 
        'lighting guidance': use_lighting_guidance, 
        'lighting pixels threshold': lighting_pixels_threshold, 
        'test image names': all_img_names, 
        'int loss weight': int_weight
    }
    combined_info = {**exp_global_config, **all_schedulers}
    combined_info_path = folder_name + '/' + 'exp_config.json'
    with open(combined_info_path, 'w') as json_file:
        json.dump(combined_info, json_file, indent=4)

    # run experiment
    for ii, img_name in enumerate(all_img_names):
        # create subfolder for each image
        subfolder_name = folder_name + '/' + 'img_' + str(ii)
        os.makedirs(subfolder_name, exist_ok=True)
        
        for k in all_schedulers.keys():
            scheduler_config = all_schedulers[k]
            init_res = scheduler_config['res_list'][0]
            for s in all_seeds:
                save_name_title = 'img_' + str(ii) + '_scheduler_' + str(k) + '_seed_' + str(s)
                save_name = subfolder_name + '/' + save_name_title
                noise_fixed = set_initial_noise(init_res, s)
                run_single_stimuli(img_name, save_name, noise_fixed, s, scheduler_config, device, int_weight, root_dir)
                print('finished img ', save_name_title)
    
    print('finished experiment.')
    