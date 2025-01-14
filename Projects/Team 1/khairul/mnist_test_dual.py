"""# Environment"""

import os
print(os.uname())

# !lscpu

output_folder = "MNIST/P100"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

model_path = os.path.join(output_folder, 'mnist_model.pth')

"""# Import Libraries"""

#import the libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import time, gc
from tqdm.auto import tqdm
from numpy import round

import crypten
# doesn't work in windows
#initialize crypten
crypten.init()
# Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)

"""# Device"""

if torch.cuda.is_available(): 
    device = torch.device('cuda')
    print(f'Cuda device {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')

print(f'Using {device} backend.')

"""# Dataset"""

from torchvision.datasets import MNIST
from torchvision import transforms

batch_size=64
test_batch_size=128
NUM_CLASSES = 10
DISABLE_PROGRESS = True

train_kwargs = {'batch_size': batch_size, 'shuffle':True}
test_kwargs = {'batch_size': test_batch_size}

transform=transforms.Compose([
    transforms.ToTensor()
])

train_data = MNIST('data', train=True, download=True, transform=transform)
test_data = MNIST('data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data,**train_kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

print(f'Shape: Train data {train_data.data.shape}, test data {test_data.data.shape}.')

"""# Model"""

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

if torch.cuda.is_available():
    model = torch.load(model_path)
else:
    model = torch.load(model_path, map_location=torch.device('cpu'))

"""# Test"""

def test_elapsed_time(model, device, test_loader, run):
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f'Test:', disable=DISABLE_PROGRESS):
            data, target = data.to(device), target.to(device)
            _ = model(data)
        
    elapsed_time = time.perf_counter() - start_time
    print(f'Run {run}, Elapsed time {elapsed_time:.4f} seconds.')
    gc.collect()
    
    return elapsed_time

RUNS = 5
print(f'\nRunning test {RUNS} times.')

test_times = []
for run in range(1, RUNS+1):
    test_times.append(
        test_elapsed_time(model, device, test_loader, run)
    )

"""# Encrypted test"""

crypten.common.serial.register_safe_class(MNISTModel)

dummy_input = torch.empty((1, 1, 28, 28)).to(device)
model_enc = crypten.nn.from_pytorch(model, dummy_input)
model_enc.encrypt().to(device)

def test_encrypted_elapsed_time(model, device, test_loader, run):
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for data, target in tqdm(test_loader, disable=DISABLE_PROGRESS):
            data = crypten.cryptensor(data)
            target = crypten.cryptensor(target)

            data, target = data.to(device), target.to(device)
            _ = model(data)
            
    elapsed_time = time.perf_counter() - start_time

    print(f'Run {run}: Elapsed time {elapsed_time:.4f} seconds.')
    gc.collect()
    return elapsed_time

print(f'\nRunning encrypted test {RUNS} times.')
encrypted_test_times = []
for run in range(1, RUNS+1):
    encrypted_test_times.append(
        test_encrypted_elapsed_time(model_enc, device, test_loader, run)
    )

from pandas import DataFrame

times_df = DataFrame({
    "Run":range(1, RUNS+1),
    "Test time":test_times,
    "Test time (encrypted)": encrypted_test_times
})

times_df.round(4).to_csv(
    os.path.join(output_folder, f'benchmark_{device.type}.csv'), 
    index=False
)