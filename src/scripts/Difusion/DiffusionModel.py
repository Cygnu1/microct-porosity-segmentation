import torch
import torch.nn as nn
import torchvision
import math
import os
import PIL
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils_synt import *
from model import *
from torch.optim import Adam

"""
Script for train diffusion model
"""

class RocksDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('L')
        img_transformed = self.transform(img)

        label = img_path.split("/")[-2]
        label = 1 if label == "coquina" else 0
        # label = 1 if label == "POSITIVE" else 0

        return img_transformed, label

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = RocksDataset(img_dir_train, mask_dir_train, data_transform, data_transform, 1.0)
    test  = RocksDataset(img_dir_test, mask_dir_test, data_transform, data_transform, 1.0)

    return torch.utils.data.ConcatDataset([train, test])
    # return train

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image), cmap="gray")
    plt.show()

def get_loss(model, x_0, t, condition):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t, condition)
    return F.mse_loss(noise, noise_pred)

def get_loss_noCondition(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t, condition):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )

    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, condition) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image(condition):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    plt.figure(figsize=(15,4))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, condition)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()

if __name__ == "__main__":
    # Define beta schedule
    T = 500
    betas = linear_beta_schedule(timesteps=T)
    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


    #### PATHS ########
    img_dir_train  = "path/for/train/"
    mask_dir_train = "path/for/train_gt/"
    img_dir_test   = "path/for/test/"
    mask_dir_test  = "path/for/test_gt/"

    
    #### HYPERPARAMS ########
    IMG_SIZE = 256
    BATCH_SIZE = 1


    ### LOAD DATASET ########
    data = load_transformed_dataset()
    print(len(data))
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for step, batch in enumerate(dataloader):
        print(batch[0][step].shape)

    ### GPU ou CPU ########
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("devcei:", device)

    train_dir = data[0]
    val_dir = data[1]
    
    train_dataloader = DataLoader(RocksDataset(train_dir), batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(RocksDataset(val_dir), batch_size=BATCH_SIZE, shuffle=True)


    ### FORWARD PROCESS: NOISE SCHEDULER #######
    # Simulate forward diffusion
    image = next(iter(train_dataloader))[0]

    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
        img, noise = forward_diffusion_sample(image, t)
        # show_tensor_image(img)


    #### MODEL ########
    model = SimpleUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    # print(model)


    #### TRAINING ########
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 500
    global_step = 0
    epoch_loss = 0

    best_loss = 1e5

    for epoch in range(epochs):
        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch+1}/{epochs}', unit='img') as pbar:    
            # for step, batch in enumerate(train_dataloader):
            for step, (patch_data, patch_label,*rest) in enumerate(train_dataloader): 
                optimizer.zero_grad()
                patch_data, patch_label = patch_data.to(device), patch_label.to(device)

                t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
                loss = get_loss(model, patch_data, t, patch_label)
                # loss = get_loss_noCondition(model, batch[0], t)
                loss.backward()
                optimizer.step()

                # if epoch % 5 == 0 and step == 0:
                #   print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

                global_step += 1
                pbar.update(1)
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model, "output/dir/for/model.pt")
                    

    # if epoch % 50 == 0:
    #     sample_plot_image(data[0][1].unsqueeze(0))

    torch.save(model, "output/dir/for/model.pt")

