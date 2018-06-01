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
    """
    def __init__(self, in_shape, in_ch, dims = 3, hidden_size = None):
        super().__init__()
        self.in_shape_2d = in_shape[0:2]
        self.depth = in_shape[2]
        self.in_ch = in_ch
        self.dims = dims
        if hidden_size is None:
            self.hidden_size = dims * self.depth # size of the final output
        else:
            self.hidden_size = hidden_size
        
        self.init_layers()
        
    def init_layers(self):
        """The architecture is img -> CNN+FC -> LSTM -> motion profile"""
        self.cnn = DnCnn(self.in_shape_2d, self.in_ch)
        self.fc = FC(self.in_shape_2d, self.in_ch, 
                     self.hidden_size, 1)
        self.lstm = LSTM(self.hidden_size, 1, self.hidden_size, 1, self.dims)
        
    def forward(self, x):
        """Input size is B x C x H x W x D"""
        self.lstm.init_hidden()
        # seq_len x B x C x hidden_size
        hidden = torch.zeros((self.depth, x.shape[0], 1, self.hidden_size))
        for d in range(self.depth):
            i = self.cnn(x[:,:,:,:,d])
            hidden[d] = self.fc(i)
        return self.lstm(hidden) # 136 x B x 3 x 1
            

class LSTM(nn.Module):
    """Implements an LSTM. 
    3d inputs are handled by flattening then reshaping.
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
        x = torch.reshape(x, (x.shape[0] * x.shape[1], self.hidden_size))
        x = self.fc(x) # seq_len*B x hidden_size
        x = torch.reshape(x, (x.shape[0], x.shape[1], self.out_ch) + 
                          self.out_shape)
        return x

    def init_hidden(self):
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

class FC(nn.Module):
    """Adds an FC layer, for example at the end of a CNN.
    3d inputs are handled by flattening then reshaping.
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

class DnCnn(nn.Module):
    """Implements the DnCNN architecture: https://arxiv.org/abs/1608.03981"""
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

class UNet(nn.Module):
    """Implements the U-Net architecture: https://arxiv.org/abs/1505.04597"""
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
        conv, shape = post_module_down(ch, ch, shape, shape // 2), shape // 2
        setattr(self, 'conv2_d0', conv)
        for i in range(1, self.depth):
            conv, ch = unet_module(ch, ch * 2, shape), ch * 2
            setattr(self, 'conv_d' + str(i), conv)
            conv, shape = post_module_down(ch, ch, shape, shape // 2), shape // 2
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
        
