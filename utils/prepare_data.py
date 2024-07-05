import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional
import os
from torchvision.transforms import v2
import random
from torchvision import tv_tensors
import omegaconf
import numpy as np
from torch.utils.data import default_collate

np.random.seed(46)
torch.manual_seed(46)


class Segmenation_dataset(Dataset):
    def __init__(self, img_list:list[str], label_list:list[str], transform: v2.Transform, phase:str):
        # as the names are in order we can just use listdir
        self.transforms = transform
        self.img_list = img_list
        self.label_list = label_list
        self.phase = phase
        self.offset = (572-388)//2
        
    @staticmethod
    def load_image(img_path:str, resize:bool=False):
        img = Image.open(img_path)
        if resize:
            img = img.resize((572, 572))
        
        img_tensor = tv_tensors.Image(img)
        # print(img_tensor.shape)
        
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
            # print(coord_img[0][0], coord_img[0][1])
            img_samples.append(img[:, coord_img[0][0]:coord_img[0][1], coord_img[1][0]:coord_img[1][1]]) #.unsqueeze(0))
            # label is still a normal 512 
            label_samples.append(label[:, coord_label[0][0]:coord_label[0][1], coord_label[1][0]:coord_label[1][1]]) #.unsqueeze(0))

        # add the batch dim, so we return [4, 1, size, size]
        # img_samples = torch.cat(img_samples, dim=0)
        # label_samples = torch.cat(label_samples, dim=0)

        return img_samples, label_samples

    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]
        
        img = Segmenation_dataset.load_image(img_path, True)
        label = Segmenation_dataset.load_image(label_path, False)
        # print(img.shape, label.shape)
        mirror_img = self.overlap_tile_strategy(img.squeeze(0), self.offset)
        
        img_samples, label_samples = self.generate_tiles(mirror_img, label)
        samples_transform = []
        label_transform = []
        for img, label in zip(img_samples, label_samples):
            img_trans, label_trans= self.transforms(img, label, self.phase)
            samples_transform.append(img_trans.unsqueeze(0))
            # the labels don't need channel dimmension so we will use it as batch
            label_transform.append(label_trans)
        
        # add the batch dim, so we return [4, 1, size, size]
        samples_transform = torch.cat(samples_transform, dim=0)
        label_transform = torch.cat(label_transform, dim=0)
        
        return samples_transform, label_transform
    

class ImageTransforms:
    def __init__(self, mean_img:float, std_img:float, ) -> None:
        self.mean_img = mean_img
        self.std_img = std_img
        
    def transform(self, image, mask, phase:str="train"):
        if phase == "train":
            # Random afine
            degrees = random.randint(0, 180)
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
            displacement = v2.ElasticTransform(alpha=65.0)._get_params(image)['displacement']
            image = v2.functional.elastic(image, displacement)
            
            torch.set_rng_state(state)
            displacement = v2.ElasticTransform(alpha=65.0)._get_params(mask)['displacement']
            mask = v2.functional.elastic(mask, displacement)


            # adjust brightness and contrast
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            image = v2.functional.adjust_brightness(image, brightness)
            image = v2.functional.adjust_contrast(image, contrast)
            mask = v2.functional.adjust_brightness(mask, brightness)
            mask = v2.functional.adjust_contrast(mask, contrast)

        # normalize
        image = image.to(torch.float32)
        mask = mask.to(torch.float32)

        # result of all the dataset
        image = v2.Normalize(mean=[self.mean_img], std=[self.std_img])(image)
        # mask = v2.Normalize(mean=[self.mean_mask], std=[self.std_mask])(mask)
        mask_th = 125

        # for the labels we need torch.long
        mask = (mask > mask_th).to(torch.long)

        
        return image, mask

        
    def __call__(self, img, label, phase:str="train"):
        return self.transform(img, label, phase)
    
def collate_fn(batch):
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
            batch[i] = batch[i].squeeze(0)

    return batch


def create_train_val_lists(data_path:str, ratio:float=0.9):
    img_path = f"{data_path}/images"
    label_path = f"{data_path}/labels"
    img_list = [os.path.join(img_path, file) for file in os.listdir(img_path)]
    label_list = [os.path.join(label_path, file) for file in os.listdir(label_path)]

    # mask = np.random.rand(len(img_list)) < 0.9

    # idx = random.sample(img_list, k=int(len(img_list)*ratio))
    idx = list(range(len(img_list)))
    random.shuffle(idx)
    
    threshold = int(ratio*len(idx))
    # print(idx, threshold)
    # print(idx[:threshold])
    
    train_img_list = [img_list[i] for i in idx[:threshold]]
    train_label_list = [label_list[i] for i in idx[:threshold]]
    
    val_img_list = [img_list[i] for i in idx[threshold:]]
    val_label_list = [label_list[i] for i in idx[threshold:]]
    
    return train_img_list, train_label_list, val_img_list, val_label_list

def prepare_data(conf:omegaconf.DictConfig):
    train_img_list, train_label_list, val_img_list, val_label_list = create_train_val_lists(conf.train.data)
    
    transforms = ImageTransforms(mean_img = 126.1648, std_img = 42.2253)
    train_data = Segmenation_dataset(train_img_list, train_label_list, transforms, "val")
    val_data = Segmenation_dataset(val_img_list, val_label_list, transforms, "val")
    
    # in the original paper they seem to use batch 1 and each batch has 4 images, as we are using overlap tile
    train_loader = DataLoader(train_data, batch_size=conf.train.batch_size, collate_fn=collate_fn, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=conf.train.batch_size, collate_fn=collate_fn, shuffle=False)
    
    return train_loader, val_loader, train_data, val_data


# def prepare_test_data(conf:omegaconf.DictConfig):
    