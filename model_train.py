import numpy as np
import argparse
import logging
import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from diffusion import *
from utils import *

class ObjectDataset(Dataset):
    def __init__(self, imgs, normals, masks=None):
        self.imgs = torch.tensor(imgs, dtype=torch.float32)
        self.normals = torch.tensor(normals, dtype=torch.float32)
        self.masks = torch.ones_like(self.imgs) if masks is None else torch.tensor(masks, dtype=torch.float32)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return {
            "image": self.imgs[index],
            "normal": self.normals[index],
            "mask": self.masks[index],
        }


def get_mask_from_normal(normal_map):
    """
    Normal ground truth is (0, 0, 0) for where the empty space. 
    Return mask for regions occupied.
    """
    b, c, h, w = normal_map.shape
    assert c == 3
    mask = (normal_map[:, 0,:,:]**2 + normal_map[:, 1,:,:]**2 + normal_map[:, 2,:,:]**2) > 0.1
    return mask


def train_model_global_diffusion(
    dataloader: DataLoader, 
    device: str, 
    diffusion_model: torch.nn.Module, 
    epochs: int = 50, 
    loss_type: str = "huber",
    scheduler : DiffusionScheduler = None,
) -> torch.nn.Module:
  
    timesteps = scheduler.timesteps
    optimizer = AdamW(diffusion_model.parameters(), lr = 2e-4)
    for epoch in range(epochs):
      for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch['image'].shape[0]
        batch_img = batch['image'].to(device).float()
        batch_normal = batch['normal'].to(device).float()

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        loss = p_losses_cond(diffusion_model, batch_normal, t, batch_img, loss_type=loss_type, scheduler=scheduler)

        if step % 100 == 0:
          logging.info(f'Epoch {epoch} - Step - {step} - Loss: {loss.item():.6f}')
        loss.backward()
        optimizer.step()
        
    return diffusion_model


def train_model_patch_diffusion(
    dataloader: DataLoader,
    device: str,
    diffusion_model: torch.nn.Module,
    patch_dim: int = 16,
    epochs: int = 50,
    loss_type: str = "huber",
    scheduler: DiffusionScheduler = None,
)-> torch.nn.Module:

    logging.info("train patch diffusion only")
    
    timesteps = scheduler.timesteps
    optimizer = AdamW(diffusion_model.parameters(), lr = 2e-4)
    
    for epoch in range(epochs):
      for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_img = batch['image'].to(device).float()
        batch_normal = batch['normal'].to(device).float()
        
        # default run: crop input 256 resolution into non-overlaping patches
        batch_img_patch = batch_patchify(batch_img, patch_dim, patch_dim)
        batch_normal_patch = batch_patchify(batch_normal, patch_dim, patch_dim)
        
        batch_size_d = batch_img_patch.shape[0]
        t = torch.randint(0, timesteps, (batch_size_d,), device=device).long()
        loss = p_losses_cond(diffusion_model, batch_normal_patch, t, batch_img_patch, loss_type=loss_type, scheduler=scheduler)
        
        if step % 100 == 0:
          logging.info(f'Epoch {epoch} - Step - {step} - Loss: {loss.item():.6f}')
        loss.backward()
        optimizer.step()

    return diffusion_model     

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train a model with specified parameters")
  parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
  parser.add_argument("--save-name", type=str, default="model", help="Suffix of the saved model")
  parser.add_argument("--loss-type", type=str, default="huber", help="loss of diffusion model: l1 l2 or huber")
  parser.add_argument("--scheduler-type", type=str, default="cosine", help="diffusion scheduler type, usually cosine or linear")
  parser.add_argument("--diffusion-timestep", type=int, default=300, help="diffusion training scheduler timestep")
  parser.add_argument("--with-position-coding", type=bool, default=True, help="concatenate positional encoding to image")
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--train-mode", type=str, default="train_d_patch")
  parser.add_argument("--dataset", type=str, default='syn5k')
  parser.add_argument("--global-imgsize", type=int, default=64)

  args = parser.parse_args()
  epochs = args.epochs; batch_size = args.batch_size; train_mode = args.train_mode

  current_datetime = datetime.datetime.now()
  timestamp = current_datetime.strftime("%Y-%m-%d")
  log_level = logging.INFO
  log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  root_dir = 'projects/sfs_diffusion_repo/'
  log_filename = root_dir + f"runs/model_training_{timestamp}_" + args.save_name + ".log"

  logging.basicConfig(filename=log_filename, level=log_level, format=log_format)

  loss_type = args.loss_type
  diffusion_scheduler = DiffusionScheduler(args.diffusion_timestep, args.scheduler_type)
  train_dataset = args.dataset

  base_dir = root_dir + 'Dataset/' + train_dataset
  all_imgs = np.load(base_dir + '_imgs.npy')
  all_nmls = np.load(base_dir + '_nmls.npy')

  device = "cuda" if torch.cuda.is_available() else "cpu"

  dataset_train = ObjectDataset(all_imgs, all_nmls)
  dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

  logging.info('Training on dataset: ' + train_dataset)
  logging.info('Model training time: ' + timestamp)
  logging.info('Diffusion training scheduler timesteps: ' + str(diffusion_scheduler.timesteps))
  logging.info('Diffusion training loss: ' + str(loss_type))
  logging.info('Training epoch: ' + str(epochs) + " batch size = " + str(batch_size))

  if train_mode == 'train_d_global':
    image_size = 256 # or 64
    channels = 3
    unet_hidden_layer = (1, 2, 4)
    image_size = args.global_imgsize
    
    diffusion_model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=unet_hidden_layer,
        condition = True,
        input_channels= 4, # grayscale image + 3 noise channel
        time_dim_coeff = 4
    )

    diffusion_model.to(device)
    logging.info('UNet hidden layer: ' + str(unet_hidden_layer))
    d = train_model_global_diffusion(dataloader, device, diffusion_model, epochs, loss_type, diffusion_scheduler)
    unet_save_dir = root_dir + 'saved_models/' + 'global_unet_' + args.train_save_name + '.ckpt'
    torch.save(d.state_dict(), unet_save_dir)

  elif train_mode == 'train_d_patch':
    unet_hidden_layer = (1, 2, 4, 8)
    image_size = 16
    channels = 3
    input_channels = 4
    
    diffusion_model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=unet_hidden_layer,
        condition = True,
        input_channels= input_channels # grayscale image + 3 noise channel
    )
    
    diffusion_model.to(device)
    logging.info('UNet hidden layer: ' + str(unet_hidden_layer))
    d = train_model_patch_diffusion(dataloader, device, diffusion_model, image_size, epochs, loss_type, diffusion_scheduler)
    unet_save_dir = root_dir + 'saved_models/' + 'patch_unet_' + args.train_save_name + '.ckpt'
    torch.save(d.state_dict(), unet_save_dir)

  else:
    print('No training mode found.')
  
  logging.info('model saved to ' + args.train_save_name)
  logging.shutdown()
