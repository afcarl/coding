import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse, json, time
import numpy as np
from modules import Net, Channel

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--n_bits', type=int, default=64)
parser.add_argument('--code_len', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_batches', type=int, default=100)
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--mid_dim', type=int, default=1024)
parser.add_argument('--bn', choices=('all', 'inside', 'no'), default='inside')
parser.add_argument('--plateau_cutoff', type=int, default=5)
parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--log', type=str, default='log')
args = parser.parse_args()
        
encoder = Net(args.n_bits, args.mid_dim, args.code_len,
              args.n_layers, args.bn, use_sigmoid=True).cuda()
decoder = Net(args.code_len, args.mid_dim, args.n_bits,
              args.n_layers, args.bn, use_sigmoid=False).cuda()
channel = Channel(args.sigma).cuda()
params = [p for p in encoder.parameters()] + [p for p in decoder.parameters()]
optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)

losses = []
n_stops = 0
n_stale_epochs = 0
n_epochs = 0
data_size = (args.n_batches, args.batch_size, args.n_bits)

while n_stops < 3:

    all_data = (torch.rand(data_size) > 0.5).type(torch.FloatTensor)
    n_correct = 0

    for data in all_data:
        optimizer.zero_grad()
        data = Variable(data.cuda())
        message = encoder(data)
        corrupted = channel(message)
        reconstruction = decoder(corrupted)
        loss = F.binary_cross_entropy_with_logits(reconstruction, data)
        loss.backward()
        optimizer.step()

        n_correct += torch.sum(((reconstruction > 0).type(torch.cuda.FloatTensor) == data).type(torch.cuda.FloatTensor))
        losses.append(loss.data[0])
        
    n_epochs += 1
    
    if losses[-args.n_batches] - np.std(losses[-args.n_batches:]) < losses[-1]:
        n_stale_epochs += 1
    else:
        n_stale_epochs = 0
        
    if n_stale_epochs > args.plateau_cutoff:
        n_stale_epochs = 0
        n_stops += 1
        optimizer.param_groups[0]['lr'] *= 0.1
        #print n_epochs
    
all_data = (torch.rand((1000, args.batch_size, args.n_bits)) > 0.5).type(torch.FloatTensor)
n_correct = 0
for data in all_data:
    data = Variable(data.cuda())
    message = encoder(data)
    corrupted = channel(message)
    reconstruction = decoder(corrupted) 
    n_correct += torch.sum(((reconstruction > 0).type(torch.cuda.FloatTensor) == data).type(torch.cuda.FloatTensor))
accuracy = float(n_correct) / all_data.nelement()
n_correct = 0
for data in all_data:
    data = Variable(data.cuda())
    message = encoder(data)
    message.data[message.data > 0] = 1
    message.data[message.data < 0] = -1
    corrupted = channel(message)
    reconstruction = decoder(corrupted) 
    n_correct += torch.sum(((reconstruction > 0).type(torch.cuda.FloatTensor) == data).type(torch.cuda.FloatTensor))
accuracy_bin = float(n_correct) / all_data.nelement()

with open(args.log, 'w') as outfile:
    json.dump({'accuracy': accuracy,
               'accuracy_bin': accuracy_bin,
               'losses': losses,
               'time': time.time() - t0},
              outfile)