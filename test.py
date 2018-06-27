import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable

from options import load_options, load_PD_dataset

from os.path import join
import sys

class NdarrayPred:
    """Loads data from NdarrayDataset and applies model.
    
    input: dataset[i] is {'image': C x H x W x D, 
                          'label': C x H x W x D}
    output: {'image': C x H x W x D, 'label': C x H x W x D,
             'pred': C x H x W x D, 'loss': 1}
    """
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        example = self.dataset[i]
        x, y = example['image'], example['label']
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        x = Variable(x).unsqueeze(0)
        y = Variable(y).unsqueeze(0)
        yp = self.net(x)
        return {'image': x[0].data.numpy(), 
                'label': y[0].data.numpy(), 
                'pred': yp[0].data.numpy()}
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class NdarrayPredSplit(NdarrayPred):
    """Applies split model to 3d data."""
    def __init__(self, dataset, net):
        super().__init__(dataset, net)
        self.shape = dataset.example['image'].shape
    
    def __len__(self):
        return len(self.dataset) // self.dataset.depth
    
    def __getitem__(self, i):
        image, label, pred = np.zeros((3,) + self.shape)
        print('Generating example of depth', self.dataset.depth, '...')
        for d in range(self.dataset.depth):
            print(d, end = ', ')
            example = super().__getitem__(i * self.dataset.depth + d)
            s = self.dataset.slice(d)
            image[s] = example['image']
            label[s] = example['label']
            pred[s] = example['pred']
        print('Done.')
            
        return {'image': image, 'label': label, 
                'pred': pred}

def view(example, d):
    image = example['image']
    label = example['label']
    pred = example['pred']
    if len(image.shape) == 3:
        s = np.index_exp[0,:,:]
    else:
        s = np.index_exp[0,:,:,d]
    image, label, pred = image[s], label[s], pred[s]
    plt.subplot(231)
    plt.imshow(image)
    plt.subplot(232)
    plt.imshow(label)
    plt.subplot(233)
    plt.imshow(pred)
    plt.subplot(234)
    plt.imshow(label-pred)
    plt.subplot(235)
    plt.imshow(image-label)
    plt.subplot(236)
    plt.imshow(image-pred)
    plt.show()

def viewpd(example, d):
    image = example['image']
    pred = example['pred']
    if len(image.shape) == 3:
        s = np.index_exp[0,:,:]
    else:
        s = np.index_exp[0,:,:,d]
    image, pred = image[s], pred[s]
    plt.subplot(131)
    plt.imshow(image)
    plt.subplot(132)
    plt.imshow(pred)
    plt.subplot(133)
    plt.imshow(image-pred)
    plt.show()

def view_losses(losses):
    plt.plot(losses[:,0], '-b')
    plt.plot(losses[:,1], '-r')
    plt.show()

def save_example(example, suffix = ''):
    np.save(join(name, 'image' + suffix), example['image'])
    np.save(join(name, 'label' + suffix), example['label'])
    np.save(join(name, 'pred' + suffix), example['pred'])

if len(sys.argv) not in (2, 3):
    print('Arguments:')
    print('1: Name of experiment (must be defined in options.py)')
    print('2 (optional): Cuda device number')
    print('Example:')
    print('python3 test.py dncnn_smallm_mag 0')
    exit()

name = sys.argv[1]
device = int(len(sys.argv) >= 3 and sys.argv[2])
save = join('../models', name)

options = load_options(name, testing = True)
train, test = options['dataset']
model = options['model']
criterion = options['criterion']
optimizer = options['optimizer']

if torch.cuda.is_available():
    torch.cuda.set_device(device)
    map_location = None
    model.cuda()
else:
    map_location = lambda storage, loc: storage
    
model.load_state_dict(torch.load(join(save, 'model.pth'), 
                                 map_location = map_location))
losses = np.load(join(save, 'losses.npy'))

# test = load_PD_dataset()
pred = NdarrayPred(test, model)

#example = pred[0]
#save_example(example, suffix = '_0')
        
