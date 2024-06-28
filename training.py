from scipy import ndimage
import numpy as np
import torch
import torch.utils
import torch.utils.data
import omegaconf
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import hydra
from utils.prepare_model import prepare_training
from utils.prepare_data import prepare_data
import os

np.random.seed(46)
torch.manual_seed(46)

# as  the unpadding conv, output dim is smaller so the input image is divided into tiles
# the optimizer is stochastic gradient descent
# loss function is cross entropy with k classes, it is pixel wise
# they use weighted loss, w = w_c + w_0 * exp(-(d_1 + d_2)^2 / 2*variance)
# with w_c being the class loss, w_0=10, d1 is distance to the border of nearest cell and d2 same to second nearest
# variance=5

# init weights using gaussian (0, sqrt(2/N)) with N the number of incoming nodes

# data aug is shift, rotation, deformations and gray value variantions, as well as random elastic deformations
# image = image * contrast_factor + brighness_factor

# data input is monochrome and labels have are 0, 1 each pixel.


def compute_weight_map(mask, w_c, w0=7.5, sigma=5):
    # Compute the distance transform
    distances = ndimage.distance_transform_edt(mask.detach().cpu().numpy() == 0)
    distances = np.clip(distances, 0, 15)
    # mask the positive values so that they have max distance, as small distance means higher loss
    distances[distances==0.0] = distances.max()

    # print(distances.shape)
    # Find the two largest distances (d1 and d2) at each pixel
    # sorted_distances = np.sort(distances.flatten())[::-1]
    # d1 = sorted_distances[0]
    # d2 = sorted_distances[1]
    # print(d1, d2)
    # Compute the weight map
    weight_map = w_c.detach().cpu().numpy() + w0 * np.exp(-((distances + distances) ** 2) / (2 * sigma ** 2))
    # print(w0 * np.exp(-((distances + distances) ** 2) / (2 * sigma ** 2)))
    return weight_map

def compute_weight_classes(mask):
    wc = torch.zeros_like(mask, dtype=torch.float32)
    class_0 = torch.sum(~mask)/(mask.shape[1]*mask.shape[2])
    class_1 = torch.sum(mask)/(mask.shape[1]*mask.shape[2])
    class_frequencies = torch.concat([class_0.unsqueeze(0), class_1.unsqueeze(0)], dim=0)
    # print(class_frequencies)
    for label, freq in enumerate(class_frequencies):
        wc[mask == label] = 1.0 / freq if freq > 0 else 0

    return wc


def train(model: torch.nn.Module, 
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            loss_fn: torch.nn.Module,
            conf: omegaconf.DictConfig,
            ):
    current_epoch = 0
    running_epochs = 0
    train_losses = 0.0
    val_losses = 0.0
    grad_norms = []
    logs = []
    
    print("Start training!")
    pbar = tqdm(total=conf.train.max_epochs)
    epochs = conf.train.max_epochs
    
    while current_epoch < epochs:
        for img, label in train_loader:
            img = img.to(conf.train.device)
            label = label.to(conf.train.device)
            
            output = model(img)
            
            # compute loss
            # as the weigth implementation is just mult, we will compute without weight and then mult
            # in this case the reduction has to be "none" so that we can multiply by the weights            
            loss = loss_fn(output, label)
            with torch.no_grad():
                wc = compute_weight_classes(label)
                weight_map = torch.from_numpy(compute_weight_map(label, wc)).to(conf.train.device)
                loss *= weight_map
                
            loss = torch.mean(loss)
            
            train_losses += loss.item()
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # compute norm
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            norm = torch.cat(grads).norm()
            grad_norms.append(norm.item())
            
            # clip grad
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()

            
        running_epochs += 1
        
        # we will do validation after the epoch ends
        if current_epoch % conf.train.eval_epoch == 0:
            with torch.no_grad():
                for img, label in val_loader:
                    img = img.to(conf.train.device)
                    label = label.to(conf.train.device)
                    
                    output = model(img)
                    
                    
                    loss = loss_fn(output, label)
                    with torch.no_grad():
                        wc = compute_weight_classes(label)
                        weight_map = torch.from_numpy(compute_weight_map(label, wc)).to(conf.train.device)
                        loss *= weight_map

                    loss = torch.mean(loss)
                    val_losses += loss.item()
        
        if current_epoch % conf.train.log_epoch == 0:
            # we want the loss per step so we divide by the num of steps that have been accumulated
            train_losses /= (len(train_loader) * running_epochs)
            val_losses /= len(val_loader)
            max_norm = max(grad_norms)
            mean_norm = sum(grad_norms)/ (len(train_loader) * running_epochs)

            print(f"Epoch {current_epoch}  || Train Loss : {train_losses} || Validation Loss : {val_losses} || Learning rate: {scheduler.state_dict()['_last_lr'][0]} || Mean Norm: {mean_norm} || Max Norm: {max_norm}" # || Trained Tokens: {total_tokens}"
                    )

            log_epoch = {'epoch': current_epoch+1, 'train_loss': train_losses, 'val_loss': val_losses,
                            "mean_norm": mean_norm, "max_norm": max_norm,
                            "learning_rate": scheduler.state_dict()['_last_lr'][0]}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv("log_output.csv", index=False)
            train_losses = 0
            val_losses = 0
            del grad_norms
            grad_norms = []
            running_epochs = 0
                
        if current_epoch % conf.train.save_epoch == 0:
            print("Saving")
            torch.save(model.state_dict(), conf.train.save_path + '/unet_' + str(current_epoch+1) + '.pth')

        # when the epoch ends we call the scheduler
        scheduler.step()
        current_epoch += 1
        pbar.update(1)
        
        
        
    print("Finished Training!")

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(conf: omegaconf.DictConfig):
    model, optim, scheduler, loss_fn = prepare_training(conf)
    
    train_loader, val_loader, train_data, val_data = prepare_data(conf)
    # each time we sample from the dataset, different transforms are applied
    # but as the dataloader stores the samples or smth so the samples don't change the transformation
    if os.path.isdir(conf.train.save_path) == False:
        os.makedirs(conf.train.save_path)
    train(model, train_loader, val_loader, optim, scheduler, loss_fn, conf)


if __name__ == "__main__":
    main()