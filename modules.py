import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, n_layers, bn_mode, use_sigmoid):
        super(Net, self).__init__()
        sizes = [[mid_dim, mid_dim] for _ in range(n_layers)]
        sizes[0][0] = in_dim
        sizes[-1][-1] = out_dim
        if bn_mode == 'no':
            use_bn = [False] * n_layers
        elif bn_mode == 'all':
            use_bn = [True] * n_layers
        elif bn_mode == 'inside':
            use_bn = [True] * (n_layers - 1) + [False]
        else:
            raise ValueError('unknown bn_mode')
        layers = OrderedDict()
        for i in range(n_layers):
            layers['fc%d' % i] = nn.Linear(sizes[i][0], sizes[i][1])
            if use_bn[i]:
                layers['bn%d' % i] = nn.BatchNorm1d(sizes[i][1])
            if i < n_layers - 1:
                layers['relu%d' % i] = nn.ReLU()
        self.use_sigmoid = use_sigmoid
        self.layers = nn.Sequential(layers)
                                                         
    def forward(self, x):
        x = self.layers(x)
        if self.use_sigmoid:
            return 2*F.sigmoid(x) - 1
        else:
            return x

class AWNGChannel(nn.Module):
    def __init__(self, p):
        super(AWNGChannel, self).__init__()
        self.p = p
        
    def forward(self, x):
        noise = self.p * torch.randn(x.size()).type(type(x.data))
        noise = Variable(noise.cuda(), requires_grad=False)
        return x + noise
    
class BSChannel(nn.Module):
    def __init__(self, p):
        super(BSChannel, self).__init__()
        self.p = p
        
    def forward(self, x):
        x.data[x.data > 0] = 1
        x.data[x.data < 0] = -1
        noise = 2 * torch.bernoulli((1 - self.p) * torch.ones(x.size())) - 1
        noise = Variable(noise.cuda(), requires_grad=False)
        return x * noise
    
class RepeatEncoder(nn.Module):
    def __init__(self, n):
        super(RepeatEncoder, self).__init__()
        self.n = n
    def forward(self, x):
        x = x.repeat(1, self.n).view(x.size(0), -1)
        return 2*x -1
        
class RepeatDecoder(nn.Module):
    def __init__(self, n):
        super(RepeatDecoder, self).__init__()
        self.n = n
    def forward(self, x):
        return x.view(x.size(0), self.n, -1).mean(dim=1)