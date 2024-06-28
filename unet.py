import torch.nn as nn
import torch
import torchvision
import omegaconf
import torchvision.transforms.functional


class conv_relu_block(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(conv_relu_block, self).__init__()
        
        self.input_channels = in_channels
        self.output_channels = out_channels
        # print(self.input_channels, self.output_channels)
        # the channels are double in the first conv
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, 
                               kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels, 
                               kernel_size=3, stride=1, padding=0)
        
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x:torch.tensor, output_activation_state:bool=False):
            
        x_activation_1 = self.activation(self.conv1(x))
        x_activation_2 = self.activation(self.conv2(x_activation_1))

        if output_activation_state:
            return x_activation_2, x_activation_1

        return x_activation_2

# contracting block
class conv_block_pooling(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(conv_block_pooling, self).__init__()
        
        self.conv_block = conv_relu_block(in_channels, out_channels)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, x: torch.tensor, output_activation_state:bool=False):
        if output_activation_state:
            x, x_activation_1 = self.conv_block(x, True)
        else:
            x = self.conv_block(x)
            
        out = self.pooling(x)
        
        if output_activation_state:
            return out, x, x_activation_1
        
        return out, x

# expansive block
class conv_block_upsample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super(conv_block_upsample, self).__init__()
        
        self.conv_block = conv_relu_block(in_channels, out_channels)
        
        # channels are reduced in the first conv
        self.upsampling = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels//2, 
                                             kernel_size=2, stride=2, padding=0)
        
    def forward(self, x: torch.tensor, output_activation_state:bool=False):
        # we assume the other layer output has already being concatenated
        if output_activation_state:
            x_activation_2, x_activation_1 = self.conv_block(x, True)
        else:
            x_activation_2 = self.conv_block(x)
            
        x = self.upsampling(x_activation_2)
        
        if output_activation_state:
            return x, x_activation_2, x_activation_1
        
        return x


class Unet(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super(Unet, self).__init__()
        self.conf = conf
        
        self.contracting = nn.ModuleList()
        self.contracting.append(conv_block_pooling(self.conf.unet.image_channels, self.conf.unet.input_channels[0]))

        for in_channels  in self.conf.unet.input_channels[:-1]:
            out_channels = in_channels*2
            self.contracting.append(conv_block_pooling(in_channels, out_channels))
        
        self.expansive = nn.ModuleList()
        self.expansive.append(conv_block_upsample(self.conf.unet.input_channels[-1], self.conf.unet.input_channels[-1]*2))

        reversed_channels = list(reversed(self.conf.unet.input_channels))
        for out_channels  in reversed_channels[:-1]:
            in_channels = int(out_channels*2)
            self.expansive.append(conv_block_upsample(in_channels, out_channels))

        self.expansive.append(conv_relu_block(reversed_channels[-1]*2, reversed_channels[-1]))
        self.final_conv = nn.Conv2d(in_channels=self.conf.unet.input_channels[0], out_channels=conf.unet.num_classes, 
                                    kernel_size=1, stride=1, padding=0)
        
        self.dropout = nn.Dropout(p=conf.unet.dropout)
        
    def crop_tensor(self, x: torch.tensor, input_size:int):
        # we have that with each convolution, the size is reduced by 2 and conv are in blocks of 2
        # in the first crop it is from 64 to 56, second is from 136 to 104. First 8=2^3 and then 32=2^5
        # in the third one is from 280 to 200  => 2^4 + 2^6. Finally 568 to 392 => 176=128 + 48
        crop_pixels = (x.shape[2] - input_size)//2
        x = x[:, :, crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
        # print(f"Size after cropping {x.shape}")
        return x
    
    def forward(self, x: torch.tensor, is_debug:bool=False):

        if is_debug:
            activation_states = []
        outputs = []
        # loop all the downsampling steps
        for i in range(len(self.contracting)):
            if is_debug:
                x, pre_out, x_activation_1 = self.contracting[i](x, True)
                activation_states.append({f"down_layer_{i}": [pre_out, x_activation_1]})
            else:
                x, pre_out = self.contracting[i](x)
            
            # print(x.shape, pre_out.shape)
            outputs.append(pre_out)
        
        # "Drop-out layers at the end of the contracting path"
        x = self.dropout(x)
        
        # start the upsampling
        for i in range(len(self.expansive)):
            if i == 0:
                # in the first upsample, there is no need to crop and copy
                if is_debug:
                    x, x_activation_2, x_activation_1 = self.expansive[i](x, True)
                    activation_states.append({f"up_layer_{i}": [x_activation_2, x_activation_1]})
                else:    
                    x = self.expansive[i](x)
            elif i != len(self.expansive)-1:
                # print(x.shape, outputs[-i].shape)
                x = torch.cat([self.crop_tensor(outputs[-i], x.shape[2]), x], dim=1)
                if is_debug:
                    x, x_activation_2, x_activation_1 = self.expansive[i](x, True)
                    activation_states.append({f"up_layer_{i}": [x_activation_2, x_activation_1]})
                else:
                    x = self.expansive[i](x)
            else:
                # we do the last stage separately as only 2 outputs in the debug case
                # print(x.shape, outputs[-i].shape)
                x = torch.cat([self.crop_tensor(outputs[-i], x.shape[2]), x], dim=1)
                if is_debug:
                    x, x_activation_1 = self.expansive[i](x, True)
                    activation_states.append({f"up_layer_{i}": [x, x_activation_1]})
                else:
                    x = self.expansive[i](x)

         
        out = self.final_conv(x)
        
        if is_debug:
            return out, x, activation_states
        
        return out
    
if __name__ == "__main__":
    path = r"config.yaml"
    conf = omegaconf.OmegaConf.load(path)
    model = Unet(conf)
    # print(model)

    import torchinfo
    
    torchinfo.summary(model, (2, 3, 572, 696), depth=5,col_names=["output_size", "num_params", "mult_adds"],)
    
    # input = torch.randn((1, 3, 572, 572))
    # out = model(input)
    # print(out.shape)