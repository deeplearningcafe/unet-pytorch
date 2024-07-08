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
    
class conv_relu_block_ln(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, tensor_size:list[int], padding:int=0) -> None:
        super(conv_relu_block_ln, self).__init__()
        
        self.input_channels = in_channels
        self.output_channels = out_channels
        # print(self.input_channels, self.output_channels)
        # the channels are double in the first conv
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.output_channels, 
                               kernel_size=3, stride=1, padding=padding)
        # self.ln1 = nn.LayerNorm([self.output_channels, *tensor_size])
        self.ln1 = nn.BatchNorm2d(self.output_channels)
        self.conv2 = nn.Conv2d(in_channels=self.output_channels, out_channels=self.output_channels, 
                               kernel_size=3, stride=1, padding=padding)
        # we have no padding so the dim becomes smaller each pass
        tensor_size = [size-2 for size in tensor_size]
        # self.ln2 = nn.LayerNorm([self.output_channels, *tensor_size])
        self.ln2 = nn.BatchNorm2d(self.output_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x:torch.tensor, output_activation_state:bool=False):
        x = self.ln1(self.conv1(x))
        x_activation_1 = self.activation(x)
        x = self.ln2(self.conv2(x_activation_1))
        x_activation_2 = self.activation(x)
        
        if output_activation_state:
            return x_activation_2, x_activation_1

        return x_activation_2


# contracting block
class conv_block_pooling(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, tensor_size:list[int]=None, padding:int=0) -> None:
        super(conv_block_pooling, self).__init__()
        
        if tensor_size != None:
            self.conv_block = conv_relu_block_ln(in_channels, out_channels, tensor_size, padding=padding)
        else:
            self.conv_block = conv_relu_block(in_channels, out_channels)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, x: torch.tensor, output_activation_state:bool=False):
        x = self.pooling(x)

        if output_activation_state:
            out, x_activation_1 = self.conv_block(x, True)
        else:
            out = self.conv_block(x)
            
        
        
        if output_activation_state:
            return out,  x_activation_1
        
        return out

# expansive block
class conv_block_upsample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, tensor_size:list[int]=None, padding:int=0) -> None:
        super(conv_block_upsample, self).__init__()
        
        if tensor_size != None:
            self.conv_block = conv_relu_block_ln(in_channels, out_channels, tensor_size, padding=padding)
        else:
            self.conv_block = conv_relu_block(in_channels, out_channels)
        
        # channels are reduced in the first conv
        self.upsampling = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                             kernel_size=2, stride=2, padding=0)
    

    def crop_tensor(self, x: torch.tensor, input_size:int):
        # we have that with each convolution, the size is reduced by 2 and conv are in blocks of 2
        # in the first crop it is from 64 to 56, second is from 136 to 104. First 8=2^3 and then 32=2^5
        # in the third one is from 280 to 200  => 2^4 + 2^6. Finally 568 to 392 => 176=128 + 48
        crop_pixels = (x.shape[2] - input_size)//2
        x = x[:, :, crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]

        return x


    def forward(self, x: torch.tensor, downsample_x:torch.tensor, output_activation_state:bool=False):

        x = self.upsampling(x)
        downsample_x = self.crop_tensor(downsample_x, x.shape[2])

        x = torch.cat([downsample_x, x], dim=1)
        # we assume the other layer output has already being concatenated
        if output_activation_state:
            x_activation_2, x_activation_1 = self.conv_block(x, True)
        else:
            x_activation_2 = self.conv_block(x)
            
        
        
        if output_activation_state:
            return x_activation_2, x_activation_1
        
        return x_activation_2

class conv_block_upsample_padding(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, tensor_size:list[int]=None, padding:int=0) -> None:
        super(conv_block_upsample_padding, self).__init__()
        
        if tensor_size != None:
            self.conv_block = conv_relu_block_ln(in_channels, out_channels, tensor_size, padding=padding)
        else:
            self.conv_block = conv_relu_block(in_channels, out_channels)
        
        # channels are reduced in the first conv
        self.upsampling = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                             kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.tensor, downsample_x:torch.tensor, output_activation_state:bool=False):

        x = self.upsampling(x)

        x = torch.cat([downsample_x, x], dim=1)
        # we assume the other layer output has already being concatenated
        if output_activation_state:
            x_activation_2, x_activation_1 = self.conv_block(x, True)
        else:
            x_activation_2 = self.conv_block(x)
            
        if output_activation_state:
            return x_activation_2, x_activation_1
        
        return x_activation_2


class Unet(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super(Unet, self).__init__()
        self.conf = conf
        
        self.contracting = nn.ModuleList()
        if self.conf.unet.use_ln:
            self.contracting.append(conv_relu_block_ln(self.conf.unet.image_channels, self.conf.unet.input_channels[0], [self.conf.unet.input_shape[0]]*2, padding=conf.unet.padding))
        else:
            self.contracting.append(conv_relu_block(self.conf.unet.image_channels, self.conf.unet.input_channels[0]))

        i = 1
        for in_channels  in self.conf.unet.input_channels:
            out_channels = in_channels*2
            if self.conf.unet.use_ln:
                self.contracting.append(conv_block_pooling(in_channels, out_channels, [self.conf.unet.input_shape[i]]*2, padding=conf.unet.padding))
            else:
                self.contracting.append(conv_block_pooling(in_channels, out_channels))
            i += 1

        self.expansive = nn.ModuleList()

        reversed_channels = list(reversed(self.conf.unet.input_channels))
        for out_channels  in reversed_channels:
            in_channels = int(out_channels*2)
            if self.conf.unet.use_ln:
                self.expansive.append(conv_block_upsample(in_channels, out_channels, [self.conf.unet.input_shape[i]]*2, padding=conf.unet.padding))
            else:
                self.expansive.append(conv_block_upsample(in_channels, out_channels))
            i += 1

        self.final_conv = nn.Conv2d(in_channels=self.conf.unet.input_channels[0], out_channels=conf.unet.num_classes, 
                                    kernel_size=1, stride=1, padding=conf.unet.padding)
        
        self.dropout = nn.Dropout(p=conf.unet.dropout)
    
    def forward(self, x: torch.tensor, is_debug:bool=False):

        if is_debug:
            activation_states = []
        outputs = []
        # loop all the downsampling steps
        for i in range(len(self.contracting)):
            if is_debug:
                x, x_activation_1 = self.contracting[i](x, True)
                activation_states.append({f"down_layer_{i}": [x, x_activation_1]})
            else:
                x = self.contracting[i](x)
            
            # the last layer output is the 1024 channel so we don't need to store it
            if i != len(self.contracting)-1:
                outputs.append(x)
        
        # "Drop-out layers at the end of the contracting path"
        x = self.dropout(x)
        
        # start the upsampling
        for i in range(len(self.expansive)):
            if is_debug:
                x, x_activation_1 = self.expansive[i](x, outputs[-(i+1)], True)
                activation_states.append({f"up_layer_{i}": [x, x_activation_1]})
            else:
                x = self.expansive[i](x, outputs[-(i+1)], False)
         
        out = self.final_conv(x)
        
        if is_debug:
            return out, x, activation_states
        
        return out

class UnetPadding(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super(UnetPadding, self).__init__()
        self.conf = conf
        
        self.contracting = nn.ModuleList()
        if self.conf.unet.use_ln:
            self.contracting.append(conv_relu_block_ln(self.conf.unet.image_channels, self.conf.unet.input_channels[0], [self.conf.unet.input_shape[0]]*2, padding=conf.unet.padding))
        else:
            self.contracting.append(conv_relu_block(self.conf.unet.image_channels, self.conf.unet.input_channels[0]))

        i = 1
        for in_channels in self.conf.unet.input_channels:
            out_channels = in_channels*2
            if self.conf.unet.use_ln:
                self.contracting.append(conv_block_pooling(in_channels, out_channels, [self.conf.unet.input_shape[i]]*2, padding=conf.unet.padding))
            else:
                self.contracting.append(conv_block_pooling(in_channels, out_channels))
            i += 1

        self.expansive = nn.ModuleList()


        reversed_channels = list(reversed(self.conf.unet.input_channels))
        reversed_inputs = list(reversed(self.conf.unet.input_shape))
        i = 0
        for out_channels  in reversed_channels:
            in_channels = int(out_channels*2)
            if self.conf.unet.use_ln:
                self.expansive.append(conv_block_upsample_padding(in_channels, out_channels, [reversed_inputs[i]]*2, padding=conf.unet.padding))
            else:
                self.expansive.append(conv_block_upsample(in_channels, out_channels))
            i += 1

        # here we need padding 0 else we add 2 new pixels to each dim
        self.final_conv = nn.Conv2d(in_channels=self.conf.unet.input_channels[0], out_channels=conf.unet.num_classes, 
                                    kernel_size=1, stride=1, padding=0)
        
        self.dropout = nn.Dropout(p=conf.unet.dropout)
    
    def forward(self, x: torch.tensor, is_debug:bool=False):

        if is_debug:
            activation_states = []
        outputs = []
        # loop all the downsampling steps
        for i in range(len(self.contracting)):
            if is_debug:
                x, x_activation_1 = self.contracting[i](x, True)
                activation_states.append({f"down_layer_{i}": [x, x_activation_1]})
            else:
                x = self.contracting[i](x)

            # the last layer output is the 1024 channel so we don't need to store it
            if i != len(self.contracting)-1:
                outputs.append(x)
        
        # "Drop-out layers at the end of the contracting path"
        x = self.dropout(x)

        # start the upsampling
        for i in range(len(self.expansive)):
            if is_debug:
                x, x_activation_1 = self.expansive[i](x, outputs[-(i+1)], True)
                activation_states.append({f"up_layer_{i}": [x, x_activation_1]})
            else:
                x = self.expansive[i](x, outputs[-(i+1)], False)
        out = self.final_conv(x)
        if is_debug:
            return out, x, activation_states
        
        return out
