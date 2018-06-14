#import gc
#import resource

import torch
import torch.nn as nn
import numpy as np

from functions import conv_padded, conv_padded_t, MultiModule, concat, double_weight_2d, tile_add

class MotionPredictor(nn.Module):
    """Predicts motion profile from motion-corrupted brain images.
    Assumes 2D planar sampling (each plane is selectively excited in the z
    dimension, then the whole xy plane is taken at once).
    Each plane has its own displacement.
    Example: If the brain is 256 x 256 x 136, then there are 136 time points
    
    in_shape - the shape of the brain (in 3D)
    in_ch - channels of the brain (example: 2 for real and imag)
    hidden_size - the shape of the CNN output to be passed into the LSTM
    
    right now it is restricted to one axis direction, otherwise you don't know
        which axis is time.
    do axis swapping in motion sim instead of here?
    """
    def __init__(self, in_shape, in_ch, dims = 3, hidden_size = None,
                 depth = 10, dropprob = 0.0):
        super().__init__()
        self.in_shape_2d = in_shape[0:2]
        self.length = in_shape[2]
        self.in_ch = in_ch
        self.dims = dims
        if hidden_size is None:
            self.hidden_size = self.length # size of the final output
        else:
            self.hidden_size = hidden_size
        self.d = 4 # amount to downsample by after cnn
        self.down_shape = [i // (2 ** self.d) for i in self.in_shape_2d]
        self.depth = depth
        self.dropprob = dropprob
        
        self.init_layers()
        
    def init_layers(self):
        """The architecture is img -> CNN+FC -> LSTM -> motion profile"""
        # self.cnn = UNet(self.in_shape_2d, self.in_ch)
        self.cnn = DnCnn(self.in_shape_2d, self.in_ch, depth = self.depth,
                         dropprob = self.dropprob)
        self.down = DownConv(self.in_shape_2d, self.in_ch, depth = self.d)
        self.fc = FC(self.down_shape, self.in_ch, self.hidden_size, self.dims)
        self.lstm = LSTM(self.hidden_size, self.dims, self.hidden_size, 
                         1, self.dims)
        
    def forward(self, x):
        """Input size is B x C x H x W x D"""
        self.lstm.init_hidden()
        # seq_len x B x C x hidden_size
        hidden = torch.zeros((self.length, x.shape[0], self.dims, 
                              self.hidden_size))
        for d in range(self.length):
            i = x[:,:,:,:,d]
            # uses extra memory each iter until it hits a certain number:
            i = self.cnn(i)
            # i = x[:,:,:,:,d]
            #gc.collect()
            #max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            #print("cnn {:.2f} MB".format(max_mem_used / 1024))
            i = self.down(i)
            #gc.collect()
            #max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            #print("down {:.2f} MB".format(max_mem_used / 1024))
            i = self.fc(i)
            hidden[d] = i
            #gc.collect()
            #max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            #print("fc {:.2f} MB".format(max_mem_used / 1024))
        x = self.lstm(hidden).permute(1, 2, 0, 3)[:,:,:,:,None]
        return x # 136 x B x 3 x 1 -> B x 3 x 256 x 1 x 1

class LSTM(nn.Module):
    """Implements an LSTM. 
    3d inputs are handled by flattening then reshaping.
    Example shapes (Seq x B x C x H x W (x D)):
    In: 136 x 1 x 3 x 136
    Out: 136 x 1 x 3 x 1
    """
    def __init__(self, in_shape, in_ch, hidden_size, out_shape, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden_size = hidden_size
        
        if isinstance(in_shape, (tuple, list, np.ndarray)):
            self.in_size = 1
            for s in in_shape:
                self.in_size *= s
            self.in_shape = in_shape
        else:
            self.in_size = in_shape
            self.in_shape = (in_shape,)
        if isinstance(out_shape, (tuple, list, np.ndarray)):
            self.out_size = 1
            for s in out_shape:
                self.out_size *= s
            self.out_shape = out_shape
        else:
            self.out_size = out_shape
            self.out_shape = (out_shape,)
        self.init_layers()
    
    def init_layers(self):
        self.lstm = nn.LSTM(self.in_size * self.in_ch, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.out_size * self.out_ch)
        self.init_hidden()
        
    def forward(self, x):
        """input shape: seq_len x B x in_ch x (in_shape)
        output shape: seq_len x B x in_ch x (out_shape)
        """
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))
        x, self.hidden = self.lstm(x, self.hidden)  # x: seq_len x B x hidden
        seq_len, B = x.shape[0], x.shape[1]
        x = torch.reshape(x, (seq_len * B, self.hidden_size))
        x = self.fc(x) # seq_len*B x hidden_size -> Seq*B x C x S
        x = torch.reshape(x, (seq_len, B, self.out_ch) + 
                          self.out_shape)
        return x

    def init_hidden(self):
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

class FC(nn.Module):
    """Adds an FC layer, for example at the end of a CNN.
    3d inputs are handled by flattening then reshaping.
    Example shapes (B x C x H x W (x D)):
    In: 1 x 2 x 32 x 32
    Out: 1 x 3 x 136
    """
    def __init__(self, in_shape, in_ch, out_shape, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        if isinstance(in_shape, (tuple, list, np.ndarray)):
            self.in_size = 1
            for s in in_shape:
                self.in_size *= s
            self.in_shape = in_shape
        else:
            self.in_size = in_shape
            self.in_shape = (in_shape,)
        if isinstance(out_shape, (tuple, list, np.ndarray)):
            self.out_size = 1
            for s in out_shape:
                self.out_size *= s
            self.out_shape = out_shape
        else:
            self.out_size = out_shape
            self.out_shape = (out_shape,)
        
        self.init_layers()
        
    def init_layers(self):
        self.fc = nn.Linear(self.in_size * self.in_ch, 
                            self.out_size * self.out_ch)
        
    def forward(self, x):
        """input shape: B x in_ch x (in_shape)
        output shape: B x out_ch x (out_shape)
        """
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc(x)
        x = torch.reshape(x, (x.shape[0], self.out_ch) + self.out_shape)
        return x

class DownConv(nn.Module):
    """Adds downsampling layers, for example between DnCnn and FC.
    Example shapes (B x C x H x W (x D)):
    In: 1 x 2 x 256 x 256
    Out: 1 x 2 x 32 x 32
    
    depth: number of factors of two to downsample by, or equivalently the # of
        downconv layers
    """
    def __init__(self, in_shape, in_ch, depth = 3):
        super().__init__()
        self.in_shape = np.array(in_shape)
        self.in_ch = in_ch
        self.depth = depth
        
        if len(in_shape) == 2:
            self.dim = '2d'
        elif len(in_shape) == 3:
            self.dim = '3d'
        else:
            assert False, 'Input ' + str(in_shape) + ' must be 2d or 3d'
        
        self.init_layers()
        
    def init_layers(self):
        shape = self.in_shape
        def post_module_down(shape, out_shape):
            conv = conv_padded(self.in_ch, self.in_ch, 2, 2, shape, out_shape, 
                               dim = self.dim)
            return MultiModule((conv, nn.PReLU(num_parameters = self.in_ch)))
        for i in range(self.depth):
            self.add_module("conv" + str(i), post_module_down(shape, shape//2))
            shape //= 2
    
    def forward(self, x):
        for i in range(self.depth):
            conv = getattr(self, "conv" + str(i))
            x = conv(x)
        return x

class DnCnn(nn.Module):
    """Implements the DnCNN architecture: https://arxiv.org/abs/1608.03981
    Example shapes (B x C x H x W (x D)):
    In: 1 x 2 x 256 x 256
    Out: 1 x 2 x 256 x 256
    
    depth: depth of network
    kernel: size of convolution kernels
    dropprob: probability for dropout
    """
    def __init__(self, in_shape, in_ch, depth = 20, kernel = 3, dropprob = 0.0):
        super().__init__()
        self.in_shape = np.array(in_shape)
        self.in_ch = in_ch
        self.depth = depth
        self.kernel = kernel
        self.dropprob = 0.0
        
        if len(in_shape) == 2:
            self.dim = '2d'
        elif len(in_shape) == 3:
            self.dim = '3d'
        else:
            assert False, 'Input ' + str(in_shape) + ' must be 2d or 3d'
        
        self.init_layers()
        
    def init_layers(self):
        """Initializes every layer of the CNN."""
        self.convi = conv_padded(self.in_ch, 64, self.kernel, 1,
            self.in_shape, self.in_shape, dim = self.dim)
        self.prelui = nn.PReLU(num_parameters = 64)
        for i in range(self.depth):
            self.add_module("conv" + str(i), conv_padded(64, 64, 
                self.kernel, 1, self.in_shape, self.in_shape, 
                dim = self.dim))
            self.add_module("post" + str(i), MultiModule(
                [nn.BatchNorm2d(64),
                 nn.PReLU(num_parameters = 64)]))
        self.convf = conv_padded(64, self.in_ch, self.kernel, 1,
            self.in_shape, self.in_shape, dim = self.dim)
        if self.dim == '2d':
            self.dropout = nn.Dropout2d(p = self.dropprob)
        else:
            self.dropout = nn.Dropout3d(p = self.dropprob)
            
    def forward(self, x):
        """Defines one forward pass given input x."""
        x = self.prelui(self.convi(x))
        for i in range(self.depth):
            conv = getattr(self, "conv" + str(i))
            post = getattr(self, "post" + str(i))
            x = post(x + conv(x))
        x = self.dropout(self.convf(x))
        return x
    
    def double(self):
        """Upsamples weights by 2 to double input shape.
        Default interp is zero-order hold.
        """
        self.in_shape *= 2
        self.kernel *= 2
        
        self.init_layers()
        dict = self.state_dict()
        keys = dict.keys()
        for key in keys:
            if key.startswith('conv') and key.endswith('weight'):
                dict[key] = double_weight_2d(dict[key])

class UAutoencoder(nn.Module):
    """Combines two UEncoders into an autoencoder.
    / is used instead of // in case shape is not divisible by 2**4. The
    conv_padded function (in functions.py) truncates decimals for shape.
    Keeping shape as a decimal means that after dividing and un-dividing, we
    get the original shape back.
    """
    def __init__(self, in_shape, in_ch, depth = 4, kernel = 3):
        super().__init__()
        self.in_shape = np.array(in_shape)
        self.in_ch = in_ch
        self.depth = depth
        self.kernel = kernel
        
        self.init_layers()
        
    def init_layers(self):
        self.encoder = UEncoder(self.in_shape, self.in_ch, depth = self.depth, 
                                kernel = self.kernel, part = 'encoder')
        self.decoder = UEncoder(self.in_shape / (2 ** self.depth), self.in_ch, 
                                depth = self.depth, kernel = self.kernel, 
                                part = 'encoder')
                
class UEncoder(nn.Module):
    """Implements an encoder or decoder based on the UNet
    architecture (https://arxiv.org/abs/1505.04597).
    Modifications (inspired by VNet):
        It uses resid learning: x = x + conv(x) instead of x = conv(x).
        It uses downconvolution instead of max pooling.
        Channels do not expand to achieve higher compression.
        No feature forwarding.
    """
    def __init__(self, in_shape, in_ch, depth = 4, kernel = 3, part = 'encoder'):
        super().__init__()
        self.in_shape = np.array(in_shape)
        self.in_ch = in_ch
        self.depth = depth
        self.kernel = kernel
        self.part = part
        
        if len(in_shape) == 2:
            self.dim = '2d'
        elif len(in_shape) == 3:
            self.dim = '3d'
        else:
            assert False, 'Input ' + str(in_shape) + ' must be 2d or 3d'
            
        self.init_layers()
        
    def init_layers(self):
        ch = self.in_ch
        shape = self.in_shape
        
        def unet_module(ch, out_ch, shape):
            # print(ch, shape)
            conv1 = conv_padded(ch, out_ch, self.kernel, 1, shape, shape, 
                                dim = self.dim)
            conv2 = conv_padded(out_ch, out_ch, self.kernel, 1, shape, shape,
                                dim = self.dim)
            return MultiModule((conv1, nn.PReLU(num_parameters = out_ch), 
                                conv2, nn.PReLU(num_parameters = out_ch)))
        def down_module(ch, out_ch, shape, out_shape):
            conv = conv_padded(ch, out_ch, 2, 2, shape, out_shape, 
                               dim = self.dim)
            return MultiModule((conv, nn.PReLU(num_parameters = out_ch)))
        def up_module(ch, out_ch, shape, out_shape):
            conv = conv_padded_t(ch, out_ch, 2, 2, shape, out_shape, 
                                 dim = self.dim)
            return MultiModule((conv, nn.PReLU(num_parameters = out_ch)))
        
        for d in range(self.depth):
            conv = unet_module(ch, ch, shape)
            setattr(self, "conv" + str(d), conv)
            if self.part == 'encoder':
                conv, shape = down_module(ch, ch, shape, shape / 2), shape / 2
            elif self.part == 'decoder':
                conv, shape = up_module(ch, ch, shape, shape * 2), shape * 2
            setattr(self, "conv_p" + str(d), conv)
            
    def forward(self, x):
        for d in range(self.depth):
            conv = getattr(self, "conv" + str(d))
            post = getattr(self, "conv_p" + str(d))
            x = post(tile_add(x, conv(x)))
        return x

class UNet(nn.Module):
    """Implements the U-Net architecture: https://arxiv.org/abs/1505.04597
    Modifications (inspired by VNet):
        It uses resid learning: x = x + conv(x) instead of x = conv(x)
        It uses downconvolution instead of max pooling.
    Example shapes (B x C x H x W (x D)):
    In: 1 x 2 x 256 x 256
    Out: 1 x 2 x 256 x 256
    
    depth: roughly half of the depth of the network (# of layers going down)
    kernel: size of convolution kernels, except the down and up convolutions,
        which are always 2x2
    dropprob: probability for dropout
    """
    def __init__(self, in_shape, in_ch, depth = 4, kernel = 3, dropprob = 0.0):
        super().__init__()
        self.in_shape = np.array(in_shape)
        self.in_ch = in_ch
        self.depth = depth
        self.kernel = kernel
        self.dropprob = dropprob
        
        if len(in_shape) == 2:
            self.dim = '2d'
        elif len(in_shape) == 3:
            self.dim = '3d'
        else:
            assert False, 'Input ' + str(in_shape) + ' must be 2d or 3d'
        
        self.init_layers()
    
    def init_layers(self):
        def unet_module(ch, out_ch, shape):
            # print(ch, shape)
            conv1 = conv_padded(ch, out_ch, self.kernel, 1, shape, shape, 
                                dim = self.dim)
            conv2 = conv_padded(out_ch, out_ch, self.kernel, 1, shape, shape,
                                dim = self.dim)
            return MultiModule((conv1, nn.PReLU(num_parameters = out_ch), 
                                conv2, nn.PReLU(num_parameters = out_ch)))
        def post_module_down(ch, out_ch, shape, out_shape):
            conv = conv_padded(ch, out_ch, 2, 2, shape, out_shape, 
                               dim = self.dim)
            return MultiModule((conv, nn.PReLU(num_parameters = out_ch)))
        def post_module_up(ch, out_ch, shape, out_shape):
            conv = conv_padded_t(ch, out_ch, 2, 2, shape, out_shape, 
                                 dim = self.dim)
            return MultiModule((conv, nn.PReLU(num_parameters = out_ch)))
        
        shape = self.in_shape
        ch = 64
        
        conv, ch = unet_module(self.in_ch, ch, shape), ch
        setattr(self, 'conv_d0', conv)
        conv, shape = post_module_down(ch, ch, shape, shape / 2), shape / 2
        setattr(self, 'conv2_d0', conv)
        for i in range(1, self.depth):
            conv, ch = unet_module(ch, ch * 2, shape), ch * 2
            setattr(self, 'conv_d' + str(i), conv)
            conv, shape = post_module_down(ch, ch, shape, shape / 2), shape / 2
            setattr(self, 'conv2_d' + str(i), conv)
        self.conv_m0, ch = unet_module(ch, ch * 2, shape), ch * 2
        # ch does not change due to feature forwarding
        self.conv2_m0, shape = post_module_up(ch, ch // 2, shape, shape * 2), shape * 2
        for i in range(0, self.depth):
            conv, ch = unet_module(ch, ch // 2, shape), ch // 2
            setattr(self, 'conv_u' + str(i), conv)
            if i != self.depth - 1: # last layer is different
                conv, shape = post_module_up(ch, ch // 2, shape, shape * 2), shape * 2
                setattr(self, 'conv2_u' + str(i), conv)
        conv = conv_padded(ch, self.in_ch, 1, 1, shape, shape, dim = self.dim) 
        setattr(self, 'conv2_u' + str(self.depth - 1), conv)
        if self.dim == '2d':
            self.dropout = nn.Dropout2d(p = self.dropprob)
        else:
            self.dropout = nn.Dropout3d(p = self.dropprob)
    
    def forward(self, x):
        features = []        
        for i in range(self.depth):
            conv = getattr(self, 'conv_d' + str(i))
            post = getattr(self, 'conv2_d' + str(i))
            x = tile_add(x, conv(x))
            features.append(x)
            x = post(x)
        x = self.conv2_m0(tile_add(x, self.conv_m0(x)))
        for i in range(self.depth):
            conv = getattr(self, 'conv_u' + str(i))
            post = getattr(self, 'conv2_u' + str(i))
            x = tile_add(x, conv(concat(x, features[self.depth - i - 1])))
            x = post(x)
        x = self.dropout(x)
        return x
        
