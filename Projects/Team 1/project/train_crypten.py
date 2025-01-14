# pip install crypten --user
import crypten

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from model import Net
import time

use_cuda = torch.cuda.is_available()
# use_mps = torch.backends.mps.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

batch_size=64
test_batch_size=1000
test_kwargs = {'batch_size': test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    test_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset = datasets.MNIST('../data', train=False,
                   transform=transform)
test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

crypten.init()
torch.set_num_threads(1)

from tqdm.notebook import tqdm
import crypten.mpc as mpc
import crypten.communicator as comm

# @mpc.run_multiprocess(world_size=2)
def get_time_elapsed_crypten(device, test_loader):
    plaintext_model = torch.load('./vanilla_pytorch_mnist_'+f'{torch.cuda.get_device_name(0)}.pth').to('cpu')
    dummy_input = torch.empty((1, 1, 28, 28))

    private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    private_model.encrypt(src=0)
    private_model.eval()
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        t0 = time.perf_counter()
        for data, target in tqdm(test_loader):
            target = target
            data_enc = crypten.cryptensor(data)
            output_enc = private_model(data_enc)
            output = output_enc.get_plain_text()
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        time_elapsed = time.perf_counter() - t0

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Time Elapsed:{}'.format(time_elapsed))
    return time_elapsed

time=get_time_elapsed_crypten(device, test_loader)

import pandas as pd
df=pd.DataFrame()
df['device']=[torch.cuda.get_device_name(0)]
df['time']=[time]
df.to_csv('crypten_time_elapsed_'+f'{torch.cuda.get_device_name(0)}.csv', index=False)
df