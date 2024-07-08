import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import v2
import random
from torchvision import tv_tensors
import omegaconf
import numpy as np
from torch.utils.data import default_collate

np.random.seed(46)
torch.manual_seed(46)


class Segmenation_dataset_cells(Dataset):
    def __init__(self, img_list:list[str], label_list:list[str]):
        # as the names are in order we can just use listdir
        self.img_list = img_list
        self.label_list = label_list
        self.offset = (572-388)//2
        
    @staticmethod
    def load_image(img_path:str, resize:bool=False):
        img = Image.open(img_path)
        if resize:
            img = img.resize((572, 572))
        
        img_tensor = tv_tensors.Image(img) / 255.0
        
        return img_tensor

    def overlap_tile_strategy(self, img, e):
        img_array = np.array(img)
        # apply mirroring, that is copy the pixels with offset and flip them
        left = np.flip(img_array[:, :e], axis=1)
        right = np.flip(img_array[:, -e:], axis=1)
        # add left and right
        new_img = np.concatenate([left, img_array, right], axis=1)
        top = np.flip(new_img[:e, :], axis=0)
        bottom = np.flip(new_img[-e:, :], axis=0)
        new_img = np.concatenate([top, new_img, bottom], axis=0)
        return torch.from_numpy(new_img).unsqueeze(0)

    
    def generate_tiles(self, img: torch.tensor, label: torch.tensor):
        # assume already in the desired resolution
        img_samples = []
        label_samples = []

        coordinates_img = [[[0, 572], [0, 572]], [[0, 572], [img.shape[1]-572, img.shape[1]]], 
                        [[img.shape[1]-572, img.shape[1]], [0, 572]], [[img.shape[1]-572, img.shape[1]], [img.shape[1]-572, img.shape[1]]]]

        # for the labels we don't need to take into account the offset
        top = label.shape[1]-388
        coordinates_label = [[[0, 388], [0, 388]], [[0, 388], [top, label.shape[1]]], 
                        [[top, label.shape[1]], [0, 388]], [[top, label.shape[1]], [top, label.shape[1]]]]

        for coord_img, coord_label in zip(coordinates_img, coordinates_label):
            img_samples.append(img[:, coord_img[0][0]:coord_img[0][1], coord_img[1][0]:coord_img[1][1]]) #.unsqueeze(0))
            # label is still a normal 512 
            label_samples.append(label[:, coord_label[0][0]:coord_label[0][1], coord_label[1][0]:coord_label[1][1]]) #.unsqueeze(0))


        return img_samples, label_samples

    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]
        
        img = Segmenation_dataset_cells.load_image(img_path, True)
        label = Segmenation_dataset_cells.load_image(label_path, False)

        mirror_img = self.overlap_tile_strategy(img.squeeze(0), self.offset)
        
        img_samples, label_samples = self.generate_tiles(mirror_img, label)
        
        # add the batch dim, so we return [4, 1, size, size]
        samples_transform = torch.cat(img_samples, dim=0)
        label_transform = torch.cat(label_samples, dim=0)
        
        return samples_transform, label_transform


class Segmenation_dataset_pedestrian(Dataset):
    def __init__(self, img_list:list[str], label_list:list[str]):
        # as the names are in order we can just use listdir
        self.img_list = img_list
        self.label_list = label_list
        self.offset = (572-388)//2
        
    @staticmethod
    def load_image(img_path:str, resize:bool=False):
        img = Image.open(img_path)
        if resize:
            img = img.resize((320, 320))
        img_tensor = tv_tensors.Image(img) / 255.0
        
        return img_tensor

    @staticmethod
    def load_label(label_path:str, resize:bool=True):
        label = Image.open(label_path)
        if resize:
            label = label.resize((320, 320))
        label_tensor = tv_tensors.Image(label)
        num_objs = torch.unique(label_tensor)[1:]
        mask = torch.zeros_like(label_tensor.squeeze(0))
        for i in range(len(num_objs)):
            mask[label_tensor.squeeze(0)==(num_objs[i])] = 1
        
        
        return mask.to(torch.long)
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]
        
        img = Segmenation_dataset_pedestrian.load_image(img_path, True)
        label = Segmenation_dataset_pedestrian.load_label(label_path, True)
        
        return img, label

class ImageTransforms:
    def __init__(self, mean_img:float, std_img:float, ) -> None:
        self.mean_img = mean_img
        self.std_img = std_img
    def transform(self, image, mask, phase:str="train"):
        for i in range(image.shape[0]):
            if phase == "train":
                # Random afine
                degrees = 0
                translate = random.uniform(0, 0.15)
                image[i] = v2.functional.affine(image[i], angle=degrees, translate=(translate, translate), scale=1.0, shear=0.0)
                mask[i] = v2.functional.affine(mask[i], angle=degrees, translate=(translate, translate), scale=1.0, shear=0.0)
    
                # RandomHorizontalFlip 
                if random.random() > 0.5:
                    image[i] = v2.functional.horizontal_flip(image[i])
                    mask[i] = v2.functional.horizontal_flip(mask[i])
    
                # ElasticTransform
                state = torch.get_rng_state()
                displacement = v2.ElasticTransform(alpha=65.0)._get_params(image[i])['displacement']
                image[i] = v2.functional.elastic(image[i], displacement)
                
                torch.set_rng_state(state)
                displacement = v2.ElasticTransform(alpha=65.0)._get_params(mask[i])['displacement']
                mask[i] = v2.functional.elastic(mask[i], displacement)
    
    
                # adjust brightness and contrast
                brightness = random.uniform(0.85, 1.15)
                contrast = random.uniform(0.85, 1.15)
                image[i] = v2.functional.adjust_brightness(image[i], brightness)
                image[i] = v2.functional.adjust_contrast(image[i], contrast)
    
        # result of all the dataset
        image = v2.Normalize(mean=[self.mean_img], std=[self.std_img])(image)
        mask_th = 0.5

        # for the labels we need torch.long
        mask = (mask > mask_th).to(torch.long)

        
        return image, mask

        
    def __call__(self, img, label, phase:str="train"):
        return self.transform(img, label, phase)
    
    
class ImageTransformsPedestrian:
    def __init__(self, mean_img:list[float], std_img:list[float], ) -> None:
        self.mean_img = mean_img
        self.std_img = std_img
    def transform(self, image, mask, phase:str="train"):
        if phase=="train":
            # Random afine
            # rotation makes the validations loss diverge so set to 0
            degrees = 0
            translate = random.uniform(0, 0.15)
            # print(degrees, translate)
            image = v2.functional.affine(image, angle=degrees, translate=(translate, translate), scale=1.0, shear=0.0)
            mask = v2.functional.affine(mask, angle=degrees, translate=(translate, translate), scale=1.0, shear=0.0)

            # RandomHorizontalFlip 
            if random.random() > 0.5:
                image = v2.functional.horizontal_flip(image)
                mask = v2.functional.horizontal_flip(mask)

            # ElasticTransform
            state = torch.get_rng_state()
            displacement = v2.ElasticTransform(alpha=40.0)._get_params(image)['displacement']
            image = v2.functional.elastic(image, displacement)
            
            torch.set_rng_state(state)
            displacement = v2.ElasticTransform(alpha=40.0)._get_params(mask)['displacement']
            mask = v2.functional.elastic(mask, displacement)


            # adjust brightness and contrast
            brightness = random.uniform(0.92, 1.12)
            contrast = random.uniform(0.92, 1.12)
            image = v2.functional.adjust_brightness(image, brightness)
            image = v2.functional.adjust_contrast(image, contrast)

        # result of all the dataset
        image = v2.Normalize(mean=self.mean_img, std=self.std_img)(image)

        
        return image, mask

        
    def __call__(self, img, label, phase:str="train"):
        return self.transform(img, label, phase)

def collate_fn(
    batch: list[np.ndarray, int],
    device: torch.device,
    transform: callable=None,
    phase: str="train") -> tuple[torch.Tensor]:
    
    batch = default_collate(batch)
    if batch[0].shape[0] > 1:
        for i in range(len(batch)):
            # [1, 4, channels, size, size]
            if len(batch[i].shape) == 5:
                batch[i] = batch[i].view(-1, batch[i].shape[2], batch[i].shape[3], batch[i].shape[4])
            else:
                # [1, 4, size, size]
                batch[i] = batch[i].view(-1, batch[i].shape[-2], batch[i].shape[-1])

    else:
        # in the case of only 1 batch then we have the 4 samples, but default_collate adds a batch dim
        for i in range(len(batch)):
            batch[i] = batch[i].transpose(0, 1)
    batch[0] = batch[0].to(device)
    batch[1] = batch[1].to(device)
    if transform is not None:
        batch[0], batch[1] = transform(batch[0], batch[1], phase)

    batch[1] = batch[1].squeeze(1)

    return batch

def collate_fn_pedestrian(
    batch: list[np.ndarray, int],
    device: torch.device,
    transform: callable=None,
    phase: str="train") -> tuple[torch.Tensor]:
    
    batch = default_collate(batch)
    batch[0] = batch[0].to(device)
    batch[1] = batch[1].to(device)
    if transform is not None:
        batch[0], batch[1] = transform(batch[0], batch[1], phase)


    return batch


def create_train_val_lists(data_path:str, ratio:float=0.9):
    img_path = f"{data_path}/images"
    label_path = f"{data_path}/labels"
    img_list = [os.path.join(img_path, file) for file in os.listdir(img_path)]
    label_list = [os.path.join(label_path, file) for file in os.listdir(label_path)]

    idx = list(range(len(img_list)))
    random.shuffle(idx)
    
    threshold = int(ratio*len(idx))
    
    train_img_list = [img_list[i] for i in idx[:threshold]]
    train_label_list = [label_list[i] for i in idx[:threshold]]
    
    val_img_list = [img_list[i] for i in idx[threshold:]]
    val_label_list = [label_list[i] for i in idx[threshold:]]
    
    return train_img_list, train_label_list, val_img_list, val_label_list

def select_dataset(conf:omegaconf.DictConfig):
    if conf.train.dataset_name == "cells":
        train_img_list, train_label_list, val_img_list, val_label_list = create_train_val_lists(conf.train.data)
        train_data = Segmenation_dataset_cells(train_img_list, train_label_list)
        val_data = Segmenation_dataset_cells(val_img_list, val_label_list)

    elif conf.train.dataset_name == "pedestrian":
        img_path = r"data\PennFudanPed\train_images"
        img_list = [os.path.join(img_path, file) for file in os.listdir(img_path)]
        label_path = r"data\PennFudanPed\train_masks"
        label_list = [os.path.join(label_path, file) for file in os.listdir(label_path)]
        train_data = Segmenation_dataset_pedestrian(img_list, label_list)
        
        img_path = r"data\PennFudanPed\valid_images"
        img_list = [os.path.join(img_path, file) for file in os.listdir(img_path)]
        label_path = r"data\PennFudanPed\valid_masks"
        label_list = [os.path.join(label_path, file) for file in os.listdir(label_path)]
        val_data = Segmenation_dataset_pedestrian(img_list, label_list)

    
    return train_data, val_data

def prepare_data(conf:omegaconf.DictConfig):
    train_data, val_data = select_dataset(conf)    
    
    if conf.train.dataset_name == "cells":
        transforms = ImageTransforms(mean_img = 0.5, std_img = 0.5)
        collate_fn = collate_fn
    elif conf.train.dataset_name == "pedestrian":
        transforms = ImageTransformsPedestrian(mean_img = [0.4910, 0.4729, 0.4497], std_img = [0.5, 0.5, 0.5])
        collate_fn = collate_fn_pedestrian

    # in the original paper they seem to use batch 1 and each batch has 4 images, as we are using overlap tile
    train_loader = DataLoader(train_data, batch_size=conf.train.batch_size, collate_fn=lambda batch: collate_fn(batch, conf.train.device, transforms, "train"), shuffle=True)
    val_loader = DataLoader(val_data, batch_size=conf.train.batch_size, collate_fn=lambda batch: collate_fn(batch, conf.train.device, transforms, "val"), shuffle=False)
    
    return train_loader, val_loader, train_data, val_data


    