import torch
import numpy as np
import torch.nn.functional as F
from utils.prepare_model import prepare_test
from utils.prepare_data import prepare_data
import hydra
import omegaconf
import utils.visualization
from training import compute_weight_classes, compute_weight_map


def debug_inference(model, batch:torch.tensor):
    output, last_hidden_state, activation_states = model(batch[0], is_debug=True)
    
    utils.visualization.plot_predictions(output, batch[1])
    print("*"*50)
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(output, batch[1])
    with torch.no_grad():
        wc = compute_weight_classes(batch[1].detach().cpu().numpy())
        weight_map = torch.from_numpy(compute_weight_map(batch[1], wc)).to("cpu")
        loss *= weight_map

    loss = torch.mean(loss)
    loss.backward()
    print(loss)
    print("*"*50)
    
    utils.visualization.plot_weight_map(weight_map)
    print(f"Min of weight map: {weight_map.min()}, Max of weight map: {weight_map.max()}")
    print("*"*50)
    
    utils.visualization.plot_activation_layer(activation_states)
    print("*"*50)
    
    layers = ["contracting.2.conv_block.conv1.weight", "contracting.3.conv_block.conv1.weight",
          "expansive.0.conv_block.conv1.weight", "expansive.1.conv_block.conv1.weight",
          "expansive.2.conv_block.conv1.weight",]
    utils.visualization.plot_gradients(layers, model)
    
    print("*"*50)
    layers = ["contracting.0.conv_block.conv1.weight", "contracting.1.conv_block.conv1.weight",
          "expansive.3.conv_block.conv1.weight", "expansive.4.conv_block.conv1.weight",
          "final_conv.weight"]
    utils.visualization.plot_gradients(layers, model)
    
    
    

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(conf:omegaconf.DictConfig):
    utils.visualization.plot_logs("logs\log_output_20240630-145822.csv")
    conf.train.device = "cpu"
    model = prepare_test(conf)
    train_loader, val_loader, train_data, val_data = prepare_data(conf)

    batch = next(iter(val_loader))
    debug_inference(model, batch)
    
if __name__ == "__main__":
    main()