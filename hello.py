import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torchvision.transforms as transforms
from generator import Generator
from discriminator import Discriminator
from utils import calculate_error, TV_loss
from torchvision.utils import make_grid
from dataset import train_loader, test_loader
from tqdm import tqdm
import torch.nn.functional as F
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

generator = Generator(1, 128)
generator = generator.to(device)
discriminator = Discriminator(1, 128)
discriminator = discriminator.to(device)

optim_G = torch.optim.Adam(generator.parameters(), lr=0.00001)
optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001)

num_epochs = 2
num_train_batches = float(len(train_loader))
num_val_batches = float(len(test_loader))

# Define constants
EPS = 1e-9
gan_weight = 1.0
l1_weight = 100.0

# CSV file setup
with open('training_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'G Adversarial Loss', 'G Reconstruction Loss', 'G Loss Total', 'D Adversarial Loss', 'PSNR', 'SSIM'])

for epoch in range(num_epochs):
    print(f"Epoch {epoch}: ", end="")
    
    G_adv_loss = 0
    G_rec_loss = 0
    G_tot_loss = 0
    D_adv_loss = 0
    
    generator.train()
    for batch, (lr, hr) in enumerate(train_loader):
        for p in discriminator.parameters():
            p.requires_grad = False
        # training generator
        optim_G.zero_grad()
        
        lr_images = lr.to(device)
        hr_images = hr.to(device)
        lr_images = lr_images.float()
        predicted_hr_images = generator(lr_images)
        predicted_hr_labels = discriminator(predicted_hr_images)

        # Adversarial loss
        gen_loss_GAN = torch.mean(-torch.log(predicted_hr_labels + EPS))
        
        # L1 loss
        gen_loss_L1 = torch.mean(torch.abs(hr_images - predicted_hr_images))
        
        # Combined generator loss
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight
    
        print(f"Loss: {gen_loss.item()}, GAN Loss: {gen_loss_GAN.item()}, L1 Loss: {gen_loss_L1.item()}")

        G_tot_loss += gen_loss.item()
        
        gen_loss.backward()
        optim_G.step()
        
        # training discriminator
        for p in discriminator.parameters():
            p.requires_grad = True
        optim_D.zero_grad()
        predicted_hr_images = generator(lr_images).detach()  # avoid back propagation to generator
        hr_images = hr_images.float()
        adv_hr_real = discriminator(hr_images)
        adv_hr_fake = discriminator(predicted_hr_images)
        df_loss = F.binary_cross_entropy_with_logits(adv_hr_real, torch.ones_like(adv_hr_real)) + F.binary_cross_entropy_with_logits(adv_hr_fake, torch.zeros_like(adv_hr_fake))
        D_adv_loss += df_loss.item()
        df_loss.backward()
        optim_D.step()

    # After each epoch, we perform validation
    with torch.inference_mode():
        val_psnr = 0
        val_ssim = 0
        for batch_idx, (lr, hr) in enumerate(test_loader):
            lr = lr.to(device)
            hr = hr.to(device)
            lr = lr.float()
            predicted_hr = generator(lr)

            psnr, ssim = calculate_error(hr, predicted_hr)
            val_psnr += psnr
            val_ssim += ssim

    val_psnr /= num_val_batches
    val_ssim /= num_val_batches
    
    # Log to CSV
    with open('training_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, G_adv_loss/num_train_batches, G_rec_loss/num_train_batches, G_tot_loss/num_train_batches, D_adv_loss/num_train_batches, val_psnr, val_ssim])
    
    print(f"PSNR: {val_psnr:.3f} SSIM: {val_ssim:.3f}\n")

    # Save the model weights every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save(generator.state_dict(), f"generator_epoch_{epoch + 1}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch + 1}.pth")

# Save the final model weights
torch.save(generator.state_dict(), "generator_final.pth")
torch.save(discriminator.state_dict(), "discriminator_final.pth")
