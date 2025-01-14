# Environment
import os
print(os.uname())

# !lscpu

output_folder = "MNIST/2080Ti"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

"""# Import Libraries"""

#import the libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import crypten
# doesn't work in windows
#initialize crypten
crypten.init()
#Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)



"""# Google drive"""

# from google.colab import drive
# drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/MyDrive/Projects/Crypten/

"""# MNIST Image Classification

## Device
"""

if torch.cuda.is_available(): 
    device = torch.device('cuda')
    print(f'Cuda device {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')

print(f'Using {device} backend.')

"""## Dataset"""

# %cd CrypTen/tutorials/

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

"""## Model"""

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

"""## Train"""

import time, gc
from tqdm.auto import tqdm
from numpy import round

def train(
    model, device, train_loader, optimizer, epoch, criterion,
    log_interval=200, dry_run=False
):
    model.train()
    progress_bar = tqdm(
        range(len(train_loader)), desc=f'Epoch {epoch} (Train)', 
        disable=DISABLE_PROGRESS
    )

    start_time = time.perf_counter()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break

        total_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix(
            loss=round(total_loss/(batch_idx+1), 4)
        )

    elapsed_time = time.perf_counter() - start_time
    print(f'\nTrain set: Average loss: {total_loss:.4f}, Elapsed time {elapsed_time:.4f} seconds.')
    gc.collect()
    return total_loss/batch_idx, elapsed_time

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f'Test:', disable=DISABLE_PROGRESS):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    elapsed_time = time.perf_counter() - start_time

    print(f'\nTest: Average loss: {test_loss:.4f}, Accuracy: {(100* accuracy):.4f}%, Elapsed time {elapsed_time:.4f} seconds.')
    gc.collect()
    return test_loss, elapsed_time, accuracy

model = MNISTModel().to(device)
learning_rate=1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
criterion = nn.CrossEntropyLoss(reduction='mean')

epochs = 10
results_columns = ['Epoch', 'Train loss', 'Test loss', 'Train time', 'Test time']
results = []
total_train_time, total_test_time = 0, 0

for epoch in range(1, epochs + 1):
    train_loss, train_time = train(model, device, train_loader, optimizer, epoch, criterion)
    test_loss, test_time, test_accuracy = test(model, device, test_loader, criterion)
    # scheduler.step()

    total_train_time += train_time
    total_test_time += test_time
    results.append([epoch, train_loss, test_loss, train_time, test_time])
    
print(f'Train time (seconds): total {total_train_time:6g}, mean {(total_train_time/epochs):6g}.\n Test time (seconds): total {total_test_time:6g}, mean {(total_test_time/epochs):6g}.')

from pandas import DataFrame
history_df = DataFrame(results, columns=results_columns)
history_df.round(4).to_csv(
    os.path.join(output_folder,'train_history.csv'), 
    index=False
)
print(history_df)

torch.save(model, os.path.join(output_folder, 'mnist_model.pth'))

"""## Benchmark"""

def test_elapsed_time(model, device, test_loader, run):
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f'Test:', disable=DISABLE_PROGRESS):
            data, target = data.to(device), target.to(device)
            output = model(data)
        
    elapsed_time = time.perf_counter() - start_time
    print(f'Run {run}, Elapsed time {elapsed_time:.4f} seconds.')
    
    return elapsed_time

RUNS = 5
print(f'\nRunning test {RUNS} times.')

test_times = []
for run in range(1, RUNS+1):
    test_times.append(
        test_elapsed_time(model, device, test_loader, run)
    )

"""# Encrypted"""

crypten.common.serial.register_safe_class(MNISTModel)
criterion = crypten.nn.CrossEntropyLoss() # Choose loss functions

dummy_input = torch.empty((1, 1, 28, 28)).to(device)
model_enc = crypten.nn.from_pytorch(model, dummy_input)
model_enc.encrypt().to(device)

"""## Test"""

def test_encrypted(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.perf_counter()
    label_eye = torch.eye(NUM_CLASSES)
    with torch.no_grad():
        for data, target in tqdm(test_loader, disable=DISABLE_PROGRESS):
            data = crypten.cryptensor(data)
            target = crypten.cryptensor(label_eye[target])

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).get_plain_text().item()  # sum up batch loss

            # evaluate accuracy
            pred = output.get_plain_text().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target = target.get_plain_text().argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    elapsed_time = time.perf_counter() - start_time

    print(f'\nEncrypted Test: Average loss: {test_loss:.4f}, Accuracy: {(100* accuracy):.4f}%, Elapsed time {elapsed_time:.4f} seconds.\n')
    gc.collect()
    return test_loss, elapsed_time, accuracy

test_loss, elapsed_time, accuracy = test_encrypted(model_enc, device, test_loader)

"""## Benchmark"""

def test_encrypted_elapsed_time(model, device, test_loader, run):
    model.eval()
    start_time = time.perf_counter()
    label_eye = torch.eye(NUM_CLASSES)
    with torch.no_grad():
        for data, target in tqdm(test_loader, disable=DISABLE_PROGRESS):
            data = crypten.cryptensor(data)
            target = crypten.cryptensor(label_eye[target])

            data, target = data.to(device), target.to(device)
            _ = model(data)
            
    elapsed_time = time.perf_counter() - start_time

    print(f'Run {run}: Elapsed time {elapsed_time:.4f} seconds.')
    gc.collect()
    return elapsed_time

print(f'Running encrypted test {RUNS} times.')
encrypted_test_times = []
for run in range(1, RUNS+1):
    encrypted_test_times.append(
        test_encrypted_elapsed_time(model_enc, device, test_loader, run)
    )

times_df = DataFrame({
    "Run":range(1, RUNS+1),
    "Test time":test_times,
    "Test time (encrypted)": encrypted_test_times
})

times_df.round(4).to_csv(
    os.path.join(output_folder, 'benchmark.csv'), 
    index=False
)