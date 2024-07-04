import torch
import omegaconf
import unet


def create_scheduler(optim:torch.optim.Optimizer, conf:omegaconf.DictConfig):
    scheduler_type = conf.train.scheduler_type
    
    if scheduler_type == "warmup-cosine":
        scheduler_warmp = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.05, end_factor=1.0, total_iters=conf.train.warmup_epochs)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=conf.train.max_epochs-conf.train.warmup_epochs, eta_min=conf.train.lr*5e-2)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[scheduler_warmp, scheduler_cosine],
                                                          milestones=[conf.train.warmup_epochs])
    elif scheduler_type == "wsd":
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.05, end_factor=1.0, total_iters=conf.train.warmup_epochs)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(conf.train.max_epochs*0.85), eta_min=conf.train.lr*5e-2)
        # we end the cosine at lr*1e-3 so we start the dacay by a factor of that lr, if we max is 2e-3 then we start the decay at 2e-6
        scheduler_decay = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1e-3, end_factor=2e-4, total_iters=int(conf.train.max_epochs*0.15))
        scheduler =  torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[scheduler_warmup, scheduler_cosine, scheduler_decay], 
                                                           milestones=[conf.train.warmup_epochs, int(conf.train.max_epochs*0.85)])
    else:
        raise "Not implemented optimizer"
        
    return scheduler

def create_optim(model: torch.nn.Module, conf:omegaconf.DictConfig):
    if conf.train.use_bitsandbytes:
        try:
            import bitsandbytes
        except:
            raise ImportError
        
        # the original paper uses sgd, but we can try to use adamw
        if conf.train.optim == 'sgd':
            optim = bitsandbytes.optim.SGD8bit(model.parameters(), lr=conf.train.lr, momentum=0.99)
        else:
            optim = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=conf.train.lr)
    else:
        if conf.train.optim == 'sgd':
            optim = torch.optim.SGD(model.parameters(), lr=conf.train.lr, momentum=0.99)
        else:
            optim = torch.optim.AdamW(model.parameters(), lr=conf.train.lr)
    
    # scheduler
    scheduler = create_scheduler(optim, conf)
    
    return optim, scheduler

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:  # バイアス項がある場合
            torch.nn.init.constant_(m.bias, 0.0)


def get_update_ratio(model, layers:dict[str, torch.tensor], diffs: dict[str, list]):
    # we have the layers dict that stores the weights and the diffs dict that stores a list with the diff in each step
    # dict are not in place so we need to create a new one
    tmp_dict = {}
    
    for key, layer in layers.items():
        for name, param in model.named_parameters():
            if key in name.split('.'):
                with torch.no_grad():
                    param_clone = param.detach().cpu().clone()
                    change = param_clone-layer
                    diffs[key].append((change.std()/param.std()).log10().item())
                    tmp_dict[key] = param_clone

    return tmp_dict, diffs

def prepare_training(conf:omegaconf.DictConfig):
    
    model = unet.Unet(conf)
    # torch.nn.init.kaiming_normal_(model.parameters(), nonlinearity='relu')
    model.apply(weights_init)
    model = model.to(conf.train.device)
    model.train()
    
    optim, scheduler = create_optim(model, conf)    
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    return model, optim, scheduler, loss_fn

def prepare_test(conf:omegaconf.DictConfig):
    
    model = unet.Unet(conf)
    net_weights = torch.load(r''.join(conf.inference.checkpoint),
                         map_location={'cuda:0': 'cpu'})
    model.load_state_dict(net_weights)
    model = model.to(conf.train.device)
    model.eval()
    return model