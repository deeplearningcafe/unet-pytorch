import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_logs(logs_path:str):
    df = pd.read_csv(r"".join(logs_path))
    print(f"Max norm: {df['max_norm'].max()}")
    print(f"Min train loss: {df['train_loss'].min()}")
    print(f"Min val loss: {df[df['val_loss']!=0.0]['val_loss'].min()}")
    
    x = np.arange(len(df["train_loss"]))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, df["train_loss"],  label='train_loss')
    ax.scatter(x, df["val_loss"], label='val_loss')
    ax.plot(x, df["mean_norm"],  label='mean_norm')
    ax.plot(x, df["max_norm"],  label='max_norm')

    ax.plot(x, df["learning_rate"]*1000, label='lr')

    ax.set(xlabel='steps', ylabel='loss and gradients',)
    ax.grid()
    ax.set_ylim(ymin=0, ymax=12)
    ax.legend()
    plt.show()
    
def plot_activation_layer(activations_array: list[dict]):
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i, dict_layer in enumerate(activations_array): # note: exclude the output layer
        for key, layer in dict_layer.items():
            for i in range(len(layer)):
                print('(%5s) activation %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (key, i, "ReLU", layer[i].mean(), layer[i].std(), (layer[i].abs() == 0.00).float().mean()*100))
                hy, hx = torch.histogram(layer[i], density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f"{key} activation {i}")
    plt.legend(legends);
    plt.title('activation distribution')
    plt.show()
    
def plot_gradients(layers_list:list[str], model:torch.nn.Module):
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []

    for name, param in model.named_parameters():
        t = param.grad
        if name in layers_list:
            print('%5s %10s | mean %+f | std %e | grad:data ratio %e' % (name, tuple(param.shape), t.mean(), t.std(), t.std() / param.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{tuple(param.shape)} {name}')
    plt.legend(legends)
    plt.title('weights gradient distribution')
    plt.show()


def plot_predictions(outputs: torch.tensor, labels:  torch.tensor, save_path:str='', epoch:int=0):

    fig = plt.figure(figsize=(12, 12))
    
    for i in range(0, 8, 2):
        output_img = torch.argmax(outputs[int(i/2)], dim=0)
        ax = fig.add_subplot(4, 2, i+1)
        shw = ax.imshow(np.array(output_img))
        bar = plt.colorbar(shw)
        
        ax = fig.add_subplot(4, 2, i+2)
        
        shw = ax.imshow(np.array(labels[int(i/2)]))
        bar = plt.colorbar(shw)
    if len(save_path) > 1:
        plt.savefig(f"{save_path}\epoch_{epoch}.png")
    else:
        plt.show()
    
def plot_weight_map(weight_map:torch.tensor):
    plt.imshow(weight_map[0], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()