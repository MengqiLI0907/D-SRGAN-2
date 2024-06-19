# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset , DataLoader
# from glob import glob
# import torchvision.transforms as transforms

# class CustomDataset(Dataset):
#     def __init__(self,lr_dir,hr_dir,transform=None):
#         self.lr_dir = lr_dir
#         self.hr_dir = hr_dir
#         self.transform = transform
        
#         self.lr_filepath = sorted(glob(os.path.join(self.lr_dir,'*.npy')))
#         self.hr_filepath = sorted(glob(os.path.join(self.hr_dir,'*.npy')))
        
#     def __len__(self):
#         return len(self.hr_filepath)
    
#     def __getitem__(self,index):
#         lr = np.load(self.lr_filepath[index])
#         hr = np.load(self.hr_filepath[index])
        
#         if self.transform:
#             lr = self.transform(lr)
#             hr = self.transform(hr)
        
#         return lr , hr


# dataset = CustomDataset(lr_dir='dataset\LR',hr_dir='dataset\HR', transform = transforms.Compose([transforms.ToTensor()]))

# train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*.80), int(len(dataset)*.20)])

# train_loader = DataLoader(train_set,batch_size=4,shuffle=True)
# test_loader = DataLoader(test_set,batch_size=4,shuffle=True)        

# import os
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from glob import glob
# import rasterio
# import torchvision.transforms as transforms

# class CustomDataset(Dataset):
#     def __init__(self, lr_dir, hr_dir, transform=None):
#         self.lr_dir = lr_dir
#         self.hr_dir = hr_dir
#         self.transform = transform

#         self.lr_filepath = sorted(glob(os.path.join(self.lr_dir, '*.tif')))
#         self.hr_filepath = sorted(glob(os.path.join(self.hr_dir, '*.tif')))
        
#         assert len(self.lr_filepath) == len(self.hr_filepath), \
#             "Mismatch between number of LR and HR files"
        
#         print(f"Found {len(self.lr_filepath)} LR files and {len(self.hr_filepath)} HR files")

#     def __len__(self):
#         return len(self.hr_filepath)

#     def __getitem__(self, index):
#         with rasterio.open(self.lr_filepath[index]) as lr_ds:
#             lr = lr_ds.read(1)  # Reading the first band
        
#         with rasterio.open(self.hr_filepath[index]) as hr_ds:
#             hr = hr_ds.read(1)  # Reading the first band

#         print(f"Loaded LR image from {self.lr_filepath[index]} with shape {lr.shape}")
#         print(f"Loaded HR image from {self.hr_filepath[index]} with shape {hr.shape}")

#         if self.transform:
#             lr = self.transform(lr)
#             hr = self.transform(hr)

#         print(f"Transformed LR image shape: {lr.shape}")
#         print(f"Transformed HR image shape: {hr.shape}")

#         return lr, hr

# # Define the transformation
# class NumpyToTensor:
#     def __call__(self, array):
#         return torch.from_numpy(array).unsqueeze(0).float()  # Add channel dimension

# transform = transforms.Compose([NumpyToTensor()])

# # Create the dataset
# dataset = CustomDataset(lr_dir='dataset/LR', hr_dir='dataset/HR', transform=transform)

# # Split the dataset into training and testing sets
# train_size = int(len(dataset) * 0.8)
# test_size = len(dataset) - train_size
# train_set, test_set = random_split(dataset, [train_size, test_size])

# # Create DataLoader
# train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=4, shuffle=True)

# # Debug prints to verify everything is set up correctly
# print(f"Number of samples in train_set: {len(train_set)}")
# print(f"Number of samples in test_set: {len(test_set)}")

# # Iterate through one batch to check shapes
# for lr, hr in train_loader:
#     print(f"LR batch shape: {lr.shape}")
#     print(f"HR batch shape: {hr.shape}")
#     break
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
import rasterio
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, lr_transform=None, hr_transform=None, hr_target_size=(128, 128)):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.hr_target_size = hr_target_size

        self.lr_filepath = sorted(glob(os.path.join(self.lr_dir, '*.tif')))
        self.hr_filepath = sorted(glob(os.path.join(self.hr_dir, '*.tif')))
        
        assert len(self.lr_filepath) == len(self.hr_filepath), \
            "Mismatch between number of LR and HR files"
        
        print(f"Found {len(self.lr_filepath)} LR files and {len(self.hr_filepath)} HR files")

    def __len__(self):
        return len(self.hr_filepath)

    def __getitem__(self, index):
        with rasterio.open(self.lr_filepath[index]) as lr_ds:
            lr = lr_ds.read(1)  # Reading the first band
        
        with rasterio.open(self.hr_filepath[index]) as hr_ds:
            hr = hr_ds.read(1)  # Reading the first band

        # Convert to tensor and resize HR image
        hr = torch.from_numpy(hr).unsqueeze(0).float()  # Add channel dimension
        hr = transforms.functional.resize(hr, self.hr_target_size)  # Resize and keep channel dimension

        # Ensure LR is also converted to tensor and has a channel dimension
        if self.lr_transform:
            lr = self.lr_transform(lr)
        
        if self.hr_transform:
            hr = self.hr_transform(hr)

        return lr, hr

# Define the transformation for LR images
class NumpyToTensor:
    def __call__(self, array):
        return torch.from_numpy(array).unsqueeze(0).float()  # Add channel dimension

lr_transform = transforms.Compose([NumpyToTensor()])
hr_transform = None  # No need for ToTensor here since HR is already a tensor

# Create the dataset
dataset = CustomDataset(lr_dir='dataset/LR', hr_dir='dataset/HR', lr_transform=lr_transform, hr_transform=hr_transform)

# Split the dataset into training and testing sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

# Create DataLoader
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=True)

# Debug prints to verify everything is set up correctly
print(f"Number of samples in train_set: {len(train_set)}")
print(f"Number of samples in test_set: {len(test_set)}")

# Iterate through one batch to check shapes
for lr, hr in train_loader:
    print(f"LR batch shape: {lr.shape}")
    print(f"HR batch shape: {hr.shape}")
    break
