# This script can check whether crypten has been installed and working properly
# import the libraries
import crypten
import torch
import torch.nn as nn
import torch.nn.functional as F

# doesn't work in windows
#initialize crypten
crypten.init()
#Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=0)
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
        
crypten.common.serial.register_safe_class(MNISTModel)
dummy_input = torch.empty((1, 1, 28, 28))
model_enc = crypten.nn.from_pytorch(MNISTModel(), dummy_input)
model_enc.encrypt()