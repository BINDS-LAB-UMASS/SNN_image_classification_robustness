import torch
from bindsnet.conversion import ann_to_snn
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from bindsnet.network.monitors import Monitor
from time import time as t_
import os
from tqdm import tqdm

percentile = 95
random_seed = 0
torch.manual_seed(random_seed)

batch_size = 32
time = 100

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset,
                                           shuffle=True)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1000)
        self.fc2 = nn.Linear(1000, 10)


    def forward(self, x):
            x = x.view(-1, 28*28)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x)

model = torch.load('trained_model.pt')






def validate():
    model.eval()
    distance, total = 0, 0
    for (data, target) in tqdm(train_loader2):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        values = output[0]
        values = values - max(values)
        values = torch.abs(values)
        distance += torch.sum(values)
        total += 9

    print("average distance:", distance/total, total)


validate()
