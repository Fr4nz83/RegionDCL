"""
@file: resnet.py
@time: 2022/09/14
This file uses the ResNet18 in the timm library.
"""
import os
import random

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py


class ImageDataset_OLD(Dataset):
    """
    This dataset is used to finetune the pretrained ResNet model
    to extract the building contour information
    """

    def __init__(self, city):
        super(ImageDataset, self).__init__()
        print('Loading image data for {}...'.format(city))
        self.images = np.load('data/processed/{}/building_raster.npz'.format(city))['arr_0']
        print('Image data loaded.')

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return self.images.shape[0]

    @staticmethod
    def collate_fn_augmentation(batch):
        result = []
        augmentations = [lambda x: x,
                         lambda x: np.flip(x, axis=0),
                         lambda x: np.flip(x, axis=1),
                         lambda x: np.rot90(x, k=1, axes=(0, 1)),
                         lambda x: np.rot90(x, k=2, axes=(0, 1)),
                         lambda x: np.rot90(x, k=3, axes=(0, 1))]
        for pic in batch:
            choice1 = random.choice(augmentations)
            choice2 = random.choice(augmentations)
            result.append(choice1(pic)[np.newaxis, :, :])
            result.append(choice2(pic)[np.newaxis, :, :])
        return np.concatenate(result, axis=0)

    @staticmethod
    def collate_fn_embed(batch):
        return np.vstack([pic[np.newaxis, :, :] for pic in batch])
        
        
class ImageDataset(Dataset):
    """
    This dataset is used to finetune the pretrained ResNet model
    to extract the building contour information

    NOTE: hdf5 file descriptors cannot be pickled when using num_workers > 1 with Pytorch's dataloaders.
          Hence, we avoid creating the file descriptor in the constructor, deferring its creation in the methods called by
          the workers (see __getitem__).
    """

    def __init__(self, images_path):
        super(ImageDataset, self).__init__()
        
        print(f'Loading raster images from {images_path}...')
        self.images_path = images_path
        with h5py.File(self.images_path, 'r') as f:
            self.length = len(f['images'])
            print(f"Shape elements in the raster image dataset: {f['images'][-1].shape}")

    def open_hdf5(self):
        self.hdf5_file = h5py.File(self.images_path, 'r')
        self.images = self.hdf5_file['images'] # if you want dataset.

    def __getitem__(self, index):
        if not hasattr(self, 'hdf5_file'):
            self.open_hdf5()
            
        # Fetch one image at a time from the HDF5 dataset.
        # Use np.array to ensure the data is read into RAM from the file
        return np.array(self.images[index])

    def __len__(self):
        return self.length


    @staticmethod
    def collate_fn_augmentation(batch):
        result = []
        augmentations = [lambda x: x,
                         lambda x: np.flip(x, axis=0),
                         lambda x: np.flip(x, axis=1),
                         lambda x: np.rot90(x, k=1, axes=(0, 1)),
                         lambda x: np.rot90(x, k=2, axes=(0, 1)),
                         lambda x: np.rot90(x, k=3, axes=(0, 1))]
        for pic in batch:
            choice1 = random.choice(augmentations)
            choice2 = random.choice(augmentations)
            result.append(choice1(pic)[np.newaxis, :, :])
            result.append(choice2(pic)[np.newaxis, :, :])
        return np.concatenate(result, axis=0)

    @staticmethod
    def collate_fn_embed(batch):
        return np.vstack([pic[np.newaxis, :, :] for pic in batch])



class ResNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ResNet, self).__init__()
        net = timm.create_model('resnet18', pretrained=True, **kwargs)
        self.net = nn.Sequential(*(list(net.children())[:-1]))
        self.projector = nn.Sequential(
            nn.Linear(net.num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64))

    def forward(self, x):
        return self.projector(self.get_feature(x))

    def get_feature(self, x):
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        return self.net(x)



class SimCLRTrainer(object):
    def __init__(self, images_path):
        self.data = ImageDataset(images_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet().to(self.device)
        self.criterion = self.infonce_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.train_loader = torch.utils.data.DataLoader(self.data,
                                                        batch_size=32,
                                                        shuffle=True,
                                                        num_workers=8,
                                                        collate_fn=ImageDataset.collate_fn_augmentation)
        self.test_loader = torch.utils.data.DataLoader(self.data,
                                                       batch_size=32,
                                                       shuffle=False,
                                                       num_workers=8,
                                                       collate_fn=ImageDataset.collate_fn_embed)

    
    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            print(f"Fine-tuning ResNet18, epoch {epoch+1}...")
            print(f"Dataset length: {len(self.data)}")
            print(f"Number of batches (Train loader): {len(self.train_loader)}")
            losses = []
            for i, batch in enumerate(tqdm(self.train_loader, desc='Processing training set')):
                # print(f"Processing batch {i}")
                x = torch.from_numpy(batch).float().to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.criterion(y_pred)
                loss.backward()
                self.optimizer.step()
                # batch.set_description(f'Epoch {epoch} loss: {loss.item()}')
                losses.append(loss.item())
            print(f'Epoch {epoch} loss: {np.mean(losses)}')


    def embed(self):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for x in self.test_loader:
                x = torch.from_numpy(x).float().to(self.device)
                embeddings.append(self.model.get_feature(x).cpu().numpy())
        return np.concatenate(embeddings, axis=0)


    def infonce_loss(self, y_pred, lamda=0.05):
        idxs = torch.arange(0, y_pred.shape[0], device=self.device)
        y_true = idxs + 1 - idxs % 2 * 2
        similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
        similarities = similarities - torch.eye(y_pred.shape[0], device=self.device) * 1e12
        similarities = similarities / lamda
        loss = F.cross_entropy(similarities, y_true)
        return torch.mean(loss)


def train_unsupervised(city):
    path_images = f"./data/processed/{city}/building_raster.hdf5"
    trainer = SimCLRTrainer(path_images)
    trainer.train(3)
    embeddings = trainer.embed()
    np.save(f'data/processed/{city}/building_features.npy', embeddings)


if __name__ == '__main__':
    os.chdir('..')
    city = 'Singapore'
    train_unsupervised(city)
