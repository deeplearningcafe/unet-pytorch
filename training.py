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
from utils.prepare_model import prepare_training, get_update_ratio
from utils.prepare_data import prepare_data
import os
from datetime import datetime
import utils.visualization

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


def compute_weight_map(mask, w_c, w0=3.5, sigma=2.0):
    # Compute the distance transform
    distances = ndimage.distance_transform_edt(mask.detach().cpu().numpy() == 0).astype(np.float32)
    distances_max = 25
    distances = np.clip(distances, 0, distances_max)
    # mask the positive values so that they have max distance, as small distance means higher loss
    distances[distances==0.0] = distances_max-19
    distances = torch.from_numpy(distances).to(w_c.device)

    # Compute the weight map
    weight_map = w_c + w0 * torch.exp(-((distances + distances) ** 2) / (2 * sigma ** 2))
    
    return weight_map

def compute_weight_classes(mask):
    batch_size = mask.shape[0]
    wc = torch.zeros_like(mask, dtype=torch.float32)
    
    for i in range(batch_size):
        sample_mask = mask[i]
        
        # Calculate the frequency of each class in the sample
        class_0 = torch.sum(sample_mask == 0) / (sample_mask.shape[0] * sample_mask.shape[1])
        class_1 = torch.sum(sample_mask == 1) / (sample_mask.shape[0] * sample_mask.shape[1])

        class_frequencies = torch.tensor([class_0, class_1], dtype=torch.float32)

        # Calculate the weights for each class
        for label, freq in enumerate(class_frequencies):
            wc[i][sample_mask == label] = 1.0 / freq if freq > 0 else 0

        # Normalize the weights so that the minimum weight is 1
        wc[i] /= wc[i].min()
    
    return wc


# def compute_weight_classes(mask):
#     # Initialize the weight map with zeros, same shape as mask
#     wc = np.zeros_like(mask, dtype=np.float32)
    
#     # Calculate class frequencies
#     class_0 = np.sum(mask == 0) / (mask.shape[0] * mask.shape[1] * mask.shape[2])
#     class_1 = np.sum(mask == 1) / (mask.shape[0] * mask.shape[1] * mask.shape[2])
#     class_frequencies = np.array([class_0, class_1], dtype=np.float32)
    
#     # Assign weights based on class frequencies
#     for label, freq in enumerate(class_frequencies):
#         wc[mask == label] = 1.0 / freq if freq > 0 else 0
    
#     return wc
# def compute_weight_classes(labels):
#     # Count the number of elements for each class
#     class_counts = torch.bincount(labels.view(-1))
#     total_count = labels.numel()

#     # Compute the frequency of each class
#     class_frequencies = class_counts / total_count

#     # Compute weights for each class
#     class_weights = 1.0 / class_frequencies

#     # Normalize weights to sum to 1
#     class_weights /= class_weights.min()

#     # Create a tensor of weights for each element in the labels tensor
#     loss_weights = torch.zeros_like(labels, dtype=torch.float32)
    
#     for i in range(len(class_weights)):
#         loss_weights[labels == i] = class_weights[i]
    
#     return loss_weights

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
    
    # val_loss_weights = []
    img_path = os.path.join(conf.train.log_path, 'imgs')
    img_path = os.path.join(img_path, f"{datetime.now().strftime(r'%Y%m%d-%H%M%S')}")
    if os.path.isdir(img_path) == False:
        os.makedirs(img_path)
        
    log_path = os.path.join(conf.train.log_path, f"log_output_{datetime.now().strftime(r'%Y%m%d-%H%M%S')}.csv")
    
    print(f"Training args: LR{conf.train.lr} || Early stopping: {conf.train.early_stopping} || w_0: {conf.train.w_0} || Warmup epochs: {conf.train.warmup_epochs}")
    # as we use shuffle True in the train loader, we can't precompute the weights map
    while current_epoch < epochs and current_epoch < conf.train.early_stopping:
        for j, (img, label) in enumerate(train_loader):
            img = img.to(conf.train.device)
            label = label.to(conf.train.device)
            
            output = model(img)
            
            # compute loss
            # as the weight implementation is just mult, we will compute without weight and then mult
            # in this case the reduction has to be "none" so that we can multiply by the weights            
            loss = loss_fn(output, label)
            # if len(train_loss_weights) == len(train_loader):
            #     loss *= train_loss_weights[j]
            # else:
            wc = compute_weight_classes(label)
            weight_map = wc
            # by default tensors don't have grad
            # weight_map = compute_weight_map(label, wc, w0=conf.train.w_0)
            loss *= weight_map
            
            loss_mean = torch.mean(loss)
            
            train_losses += loss_mean.item()

            # backward
            optimizer.zero_grad()
            loss_mean.backward()
            
            # compute norm
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            norm = torch.cat(grads).norm()
            grad_norms.append(norm.item())
            
            # clip grad
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
            # save the first sample of the batch as will have the desired shape
            if j == 0:
                utils.visualization.plot_predictions(output.detach().cpu().numpy(), label.detach().cpu().numpy(), save_path=img_path, epoch=current_epoch, phase="train")
                utils.visualization.plot_loss_weights(weight_map.detach().cpu().numpy(), loss.detach().cpu().numpy(), save_path=img_path, epoch=current_epoch, phase="train")

        running_epochs += 1
        
        # we will do validation after the epoch ends
        if current_epoch % conf.train.eval_epoch == 0:
            with torch.no_grad():
                for i, (img, label) in enumerate(val_loader):
                    img = img.to(conf.train.device)
                    label = label.to(conf.train.device)
                    
                    output = model(img)
                    
                    # the validation samples are always the same and in the same order so we can store the loss weights
                    loss = loss_fn(output, label)
                    # if len(val_loss_weights) == len(val_loader):
                    #     weight_map = val_loss_weights[i]
                    #     loss *= weight_map
                    # else:
                    # label_cpu = label.detach().cpu().numpy()
                    wc = compute_weight_classes(label)
                    weight_map = wc
                    # weight_map = compute_weight_map(label, wc, w0=conf.train.w_0)
                    # weight_map = torch.from_numpy(weight_map_numpy).to(conf.train.device)
                    # val_loss_weights.append(weight_map)
                    loss *= weight_map

                    loss_mean = torch.mean(loss)
                    val_losses += loss_mean.item()

                    # save the first batch as we are sure it has all the samples
                    if i == 0:
                        # save outputs img, of the last batch of validation
                        utils.visualization.plot_predictions(output.detach().cpu().numpy(), label.detach().cpu().numpy(), save_path=img_path, epoch=current_epoch)
                        utils.visualization.plot_loss_weights(weight_map.detach().cpu().numpy(), loss.detach().cpu().numpy(), save_path=img_path, epoch=current_epoch)
        
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
            df.to_csv(log_path, index=False)
            train_losses = 0
            val_losses = 0
            del grad_norms
            grad_norms = []
            running_epochs = 0
                
        if current_epoch % conf.train.save_epoch == 0 or current_epoch == epochs:
            print("Saving")
            torch.save(model.state_dict(), conf.train.save_path + '/unet_' + str(current_epoch+1) + '.pth')

        # when the epoch ends we call the scheduler
        scheduler.step()
        current_epoch += 1
        pbar.update(1)
        
    print("Finished Training!")
    return logs

def overfit_one_batch(model, batch, optim, scheduler, loss_fn, conf: omegaconf.DictConfig, output_log:bool=True, save_update_ratio:bool=False):
    img = batch[0].to(conf.train.device)
    label = batch[1].to(conf.train.device)
    losses = []
    grad_norms = []
    current_step = 0
    logs = {}
    lrs = []
    pbar = tqdm(total=conf.overfit_one_batch.max_steps)
    
    img_path = os.path.join(conf.train.log_path, 'imgs')
    img_path = os.path.join(img_path, f"{datetime.now().strftime(r'%Y%m%d-%H%M%S')}")
    if os.path.isdir(img_path) == False:
        os.makedirs(img_path)
        
    log_path = os.path.join(conf.train.log_path, f"log_output_{datetime.now().strftime(r'%Y%m%d-%H%M%S')}.csv")


    if save_update_ratio:
        diffs = {"final_conv": []}
        layers = {"final_conv": model.final_conv.weight.detach().cpu().clone()}
    
    # we just need to compute this one time
    wc = compute_weight_classes(label)
    weight_map = compute_weight_map(label, wc, w0=conf.train.w_0)
    # weight_map = torch.from_numpy(weight_map_numpy).to(conf.train.device)

    print("Start overfitting in one batch!")
    while current_step < conf.overfit_one_batch.max_steps:
        output = model(img)
        
        loss = loss_fn(output, label)
        loss *= weight_map
            
        loss_mean = torch.mean(loss)
        losses.append(loss_mean.item())
        
        # backward
        optim.zero_grad()
        loss_mean.backward()
        
        # compute norm
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        grad_norms.append(norm.item())
        lrs.append(scheduler.state_dict()['_last_lr'][0])
        
        # clip grad
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optim.step()
        scheduler.step()

        if save_update_ratio:
            # update the change ratio
            layers, diffs = get_update_ratio(model, layers, diffs)

        current_step += 1
        pbar.update(1)
        if current_step > 1 and abs(losses[-2]-losses[-1]) < conf.overfit_one_batch.tolerance:
            break

        if current_step % conf.overfit_one_batch.logging_steps == 0:
            utils.visualization.plot_predictions(output.detach().cpu().numpy(), label.detach().cpu().numpy(), save_path=img_path, epoch=current_step)
            utils.visualization.plot_loss_weights(weight_map.detach().cpu().numpy(), loss.detach().cpu().numpy(), save_path=img_path, epoch=current_step)

            print(f"Step {current_step}  || Loss : {losses[-1]} || Learning rate: {lrs[-1]} || Norm: {grad_norms[-1]}")


    logs = {'losses': losses, "gradient_norm": grad_norms, "learning_rate": lrs}
    if output_log:
        df = pd.DataFrame(logs)
        df.to_csv(log_path, index=False)
    if save_update_ratio:
        return logs, diffs
    return logs


def grid_search(train_loader, val_loader, conf:omegaconf.DictConfig):
    logs = {'losses': [], "gradient_norm": [], "learning_rate": []}
    parameters = ["warmup", "max_lr", "w_0"]

    param_list = []
    warmup_list = [10*factor for factor in range(2, 3)]
    lr_list = [1e-4]
    w_0_list = [2.0, 2.5, 3.0, 3.5, 4.0]
    
    for i in range(len(warmup_list)):
        for j in range(len(lr_list)):
            for k in range(len(w_0_list)):
                warmup_steps = int(conf.train.max_epochs*(warmup_list[i]/100))
                param_list.append([warmup_steps, lr_list[j], w_0_list[k]])
    # parameters.append("max_lr")
    
    
    for param in parameters:
        logs[param] = []
        
    conf.unet.input_channels[:] = [channel//2 for channel in conf.unet.input_channels]
    for param in param_list:
        conf.train.warmup_epochs = int(param[0])
        text = f"Evaluating warmup steps: {conf.train.warmup_epochs}"

        # if len(param) == 2:
        conf.train.lr = param[1]
        text += f" and max_lr of {conf.train.lr}"
        
        conf.train.w_0 = param[2]
        text += f" and w_0 of {param[2]}"
        print(text)
        

        model, optim, scheduler, loss_fn = prepare_training(conf)
        if conf.overfit_one_batch.full_train:
            logs_one_comb = train(model, train_loader, val_loader, optim, scheduler, loss_fn, conf)
        else:
            batch = next(iter(train_loader))
            logs_one_comb = overfit_one_batch(model, batch, optim, scheduler, loss_fn, conf, output_log=True, save_update_ratio=False)
        
        logs_one_comb["warmup"] = [int(param[0])] * len(logs_one_comb["losses"])
        # if len(param) == 2:
        logs_one_comb["max_lr"] = [param[1]] * len(logs_one_comb["losses"])
        logs_one_comb["w_0"] = [param[2]] * len(logs_one_comb["losses"])

        for key in logs_one_comb.keys():
            logs[key].extend(logs_one_comb[key])
        
    
    df = pd.DataFrame(logs)
    df.to_csv(f"{conf.train.log_path}\hp_search_{datetime.now().strftime(r'%Y%m%d-%H%M%S')}.csv", index=False)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(conf: omegaconf.DictConfig):
    # delete this lane if want to use original model
    conf.unet.input_channels[:] = [channel//2 for channel in conf.unet.input_channels]
    
    model, optim, scheduler, loss_fn = prepare_training(conf)

    train_loader, val_loader, train_data, val_data = prepare_data(conf)
    
    if conf.overfit_one_batch.hp_search:
        grid_search(train_loader, val_loader, conf)
    elif conf.overfit_one_batch.overfit:
        batch = next(iter(train_loader))
        overfit_one_batch(model, batch, optim, scheduler, loss_fn, conf, output_log=True, save_update_ratio=True)

    else:
        # each time we sample from the dataset, different transforms are applied
        # and each time we loop the dataloder we get different transforms
        conf.train.save_path = os.path.join(conf.train.save_path, f"{datetime.now().strftime(r'%Y%m%d-%H%M%S')}")
        if os.path.isdir(conf.train.save_path) == False:
            os.makedirs(conf.train.save_path)
        if os.path.isdir(conf.train.log_path) == False:
            os.makedirs(conf.train.log_path)
        img_log_path = os.path.join(conf.train.log_path, 'imgs')
        if os.path.isdir(img_log_path) == False:
            os.makedirs(img_log_path)
            
        train(model, train_loader, val_loader, optim, scheduler, loss_fn, conf)


if __name__ == "__main__":
    main()